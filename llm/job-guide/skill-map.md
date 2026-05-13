---
layout: post
title: "岗位能力地图"
track: "🤖 大模型"
---

# 岗位能力地图 — 对照自查，精准补全短板

> 按岗位方向勾选你已掌握的技能，找出短板，按优先级补全。

---

## 应用开发工程师 能力清单

### 编程基础
- [ ] Python 异步编程（asyncio、aiohttp、httpx）
- [ ] 类型注解与 Pydantic 数据验证
- [ ] FastAPI 构建 RESTful API
- [ ] PostgreSQL / MySQL 数据库操作
- [ ] Redis 缓存与消息队列

### 工程工具
- [ ] Docker 容器化打包与运行
- [ ] Docker Compose 多服务编排
- [ ] Git 版本控制与协作
- [ ] K8s 基础（Pod/Service/Deployment）
- [ ] CI/CD 基础（GitHub Actions）

### 大模型调用
- [ ] OpenAI/DeepSeek/通义千问 API 调用
- [ ] 本地部署开源模型（Ollama/vLLM）
- [ ] 流式输出（SSE/WebSocket）
- [ ] 结构化输出（JSON Mode/Function Calling）
- [ ] Prompt 工程（CoT/少样本/角色设定）

### RAG系统
- [ ] 文档解析（PDF/Word/Markdown/网页）
- [ ] 语义分块（递归分块/语义分块）
- [ ] Embedding 模型选型（BGE/Sentence-BERT）
- [ ] 向量数据库操作（Chroma/Milvus/pgvector）
- [ ] 混合检索（BM25 + 向量）
- [ ] 重排序（Reranker）
- [ ] 幻觉检测与抑制
- [ ] RAG 评估（召回率/精准率/事实一致性）

### Agent开发
- [ ] LangChain Agent 基础
- [ ] LangGraph 工作流编排
- [ ] 工具调用（Function Calling）
- [ ] 记忆管理（短期/长期/向量记忆）
- [ ] 任务规划（CoT/ReAct）
- [ ] 多 Agent 协作（CrewAI/AutoGen）
- [ ] 错误处理与重试机制

### Harness工程化
- [ ] 系统指令层（System Prompt 规范设计）
- [ ] 工具总线搭建
- [ ] 记忆持久化（Redis + pgvector）
- [ ] 沙箱安全（Docker 隔离代码执行）
- [ ] 链路追踪（OpenTelemetry）
- [ ] 监控告警（W&B/MLflow）

---

## 算法工程师 能力清单

### 数学基础
- [ ] 线性代数（矩阵运算、特征值、SVD）
- [ ] 概率统计（贝叶斯、KL散度、交叉熵）
- [ ] 优化理论（梯度下降、Adam、学习率调度）

### 深度学习
- [ ] PyTorch 熟练（Tensor/Autograd/训练循环）
- [ ] Transformer 架构深度理解
- [ ] 注意力机制（MHA/MQA/GQA）
- [ ] 位置编码（RoPE/ALiBi）
- [ ] MoE（混合专家）架构

### 模型训练
- [ ] HuggingFace Transformers 使用
- [ ] HuggingFace PEFT 微调框架
- [ ] LoRA 原理与实现
- [ ] QLoRA（4bit量化+LoRA）
- [ ] 指令数据集构建与清洗
- [ ] SFT 监督微调全流程
- [ ] DPO 偏好对齐
- [ ] GRPO/RLHF 强化学习对齐

### 推理优化
- [ ] vLLM 部署与配置
- [ ] KV Cache 原理与优化
- [ ] 量化（GPTQ/AWQ/INT4/INT8）
- [ ] FlashAttention
- [ ] 投机解码（Speculative Decoding）
- [ ] 分布式推理（张量并行/流水线并行）

### 模型评测
- [ ] Perplexity 困惑度计算
- [ ] BLEU/ROUGE 评估
- [ ] 事实一致性评估
- [ ] 大模型评测基准（MMLU/HumanEval/MT-Bench）

---

## 系统/LLMOps工程师 能力清单

### 基础设施
- [ ] K8s 集群管理与服务编排
- [ ] GPU 集群管理（NVIDIA DCGM）
- [ ] 分布式存储（对象存储/NFS）
- [ ] 网络配置（InfiniBand/RDMA）

### MLOps工具链
- [ ] W&B（实验跟踪/模型监控）
- [ ] MLflow（实验管理/模型仓库）
- [ ] DVC（数据版本控制）
- [ ] Airflow/Prefect（工作流调度）

### 模型服务化
- [ ] vLLM/SGLang 生产级推理服务
- [ ] Triton Inference Server
- [ ] 负载均衡与高可用
- [ ] API 限流与熔断
- [ ] 成本优化（Spot实例/批处理）

### 监控与可观测性
- [ ] Prometheus + Grafana
- [ ] OpenTelemetry 链路追踪
- [ ] 日志聚合（ELK/Loki）
- [ ] 幻觉检测自动化
- [ ] 性能基准测试（吞吐量/延迟/TTFT）

---

## 优先级建议

**新手入门（0基础）**：Python → Docker → LLM API调用 → 简单RAG → LangChain Agent

**有Python基础**：FastAPI → RAG全流程 → LangGraph → LoRA微调 → vLLM

**有深度学习基础**：QLoRA微调 → DPO对齐 → vLLM优化 → Harness架构 → LLMOps

---

[← 返回求职指南](./index) | [→ Python工程栈](./python-engineering)
