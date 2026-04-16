# AI Infra 学习路线图（52周）

> 从 GPU 基础到大规模训练系统，每周一个主题，循序渐进，系统成长。

---

## 阶段一：基础夯实（Week 1-12）

### 🔷 Module 1：GPU 架构与 CUDA 编程（Week 1-4）

| 周次 | 主题 | 关键内容 |
|------|------|---------|
| **Week 1** | GPU架构与CUDA基础 | GPU vs CPU 哲学差异 / SM架构与Warp调度 / CUDA编程模型 / 矩阵转置 / 归约优化 / Nsight工具 / SGEMM实战 |
| **Week 2** | CUDA 进阶优化技术 | Tensor Core与wmma API / FP16混合精度 / 流水线（Software Pipelining）/ 异步内存拷贝（cp.async）/ CUDA Graph / 多Stream并发 |
| **Week 3** | cuBLAS / cuDNN 深度解析 | cuBLAS API与性能调优 / cuDNN卷积算法选择 / cuDNN Workspace管理 / 算子融合（Op Fusion）基础 / NHWC vs NCHW格式 |
| **Week 4** | CUDA 内核编写综合实战 | Softmax CUDA Kernel（Online Safe Softmax）/ LayerNorm Kernel / FlashAttention原理与核心代码解析 / Triton入门 |

---

### 🔷 Module 2：深度学习框架原理（Week 5-8）

| 周次 | 主题 | 关键内容 |
|------|------|---------|
| **Week 5** | PyTorch 核心架构 | Autograd机制与计算图 / Tensor存储与视图 / 算子注册（Register Operator）/ Dispatcher机制 / DispatchKey |
| **Week 6** | PyTorch 算子开发 | 自定义CUDA Extension / torch.autograd.Function / cpp_extension编写 / 算子性能测试与验证 |
| **Week 7** | PyTorch 内存管理 | CachingAllocator实现 / 显存碎片化问题 / `torch.cuda.memory_stats` 分析 / Gradient Checkpointing原理与实现 |
| **Week 8** | PyTorch 图优化 | TorchScript与FX图 / torch.compile（Dynamo/Inductor）/ 算子融合（Pattern Matching）/ AOT Autograd |

---

### 🔷 Module 3：分布式基础（Week 9-12）

| 周次 | 主题 | 关键内容 |
|------|------|---------|
| **Week 9** | 分布式通信基础 | 集合通信原语（AllReduce/Broadcast/AllGather/ReduceScatter）/ Ring-AllReduce算法 / NCCL原理 / 带宽与延迟分析 |
| **Week 10** | 数据并行训练（DP） | DDP原理与实现 / Gradient Bucketing / 通信与计算重叠 / `torch.distributed` API / ZeRO-0/1/2 |
| **Week 11** | 模型并行基础（MP） | 张量并行（Tensor Parallelism）/ Megatron-LM列行切分 / 流水线并行（Pipeline Parallelism）/ GPipe/PipeDream |
| **Week 12** | 混合并行实战 | 3D并行（DP+TP+PP）/ FSDP（ZeRO-3）/ 显存估算模型 / 并行策略选择方法论 |

---

## 阶段二：系统深化（Week 13-26）

### 🔷 Module 4：大模型训练工程（Week 13-18）

| 周次 | 主题 | 关键内容 |
|------|------|---------|
| **Week 13** | 大规模训练框架解析 | Megatron-LM代码结构 / 混合并行实现细节 / 序列并行（Sequence Parallelism）/ Selective Activation Recomputation |
| **Week 14** | 混合精度训练 | FP16/BF16训练原理 / Loss Scaling / Master Weight / 数值稳定性问题 / AMP（torch.amp）API |
| **Week 15** | 训练稳定性与调试 | 梯度爆炸/消失诊断 / 梯度裁剪 / Loss Spike分析 / 训练过程异常检测 / Fault Tolerance基础 |
| **Week 16** | Checkpointing 与恢复 | 分布式Checkpoint（ShardedCheckpoint）/ 快速Checkpoint策略 / 断点续训 / DeepSpeed Save/Load |
| **Week 17** | 通信优化技术 | Overlap通信与计算 / 梯度压缩（Gradient Compression）/ 异步SGD / NVLink与InfiniBand网络拓扑 |
| **Week 18** | 数据流水线优化 | 大规模数据集管理 / WebDataset / 数据预处理并行化 / 数据增强流水线 / MosaicML Streaming |

---

### 🔷 Module 5：推理引擎与优化（Week 19-22）

| 周次 | 主题 | 关键内容 |
|------|------|---------|
| **Week 19** | 推理框架概览 | TensorRT原理与流程 / ONNX生态 / vLLM架构 / 推理吞吐量vs延迟权衡 |
| **Week 20** | 量化技术 | PTQ（训练后量化）/ QAT（量化感知训练）/ INT8/INT4量化 / AWQ/GPTQ算法 / KV Cache量化 |
| **Week 21** | KV Cache 与推理加速 | KV Cache原理与显存估算 / PagedAttention（vLLM）/ 连续批处理（Continuous Batching）/ Speculative Decoding |
| **Week 22** | 算子融合与编译优化 | TVM基础 / Triton编译器原理 / Kernel Fusion策略 / torch.compile推理优化 / TensorRT Plugin开发 |

---

### 🔷 Module 6：集群基础设施（Week 23-26）

| 周次 | 主题 | 关键内容 |
|------|------|---------|
| **Week 23** | GPU 集群架构 | DGX系统架构 / NVLink拓扑（NVSwitch）/ InfiniBand HDR/NDR / 集群网络设计 / RDMA基础 |
| **Week 24** | 作业调度系统 | Kubernetes+GPU / Slurm工作负载管理 / 多租户资源隔离 / 优先级调度 / MIG（Multi-Instance GPU）|
| **Week 25** | 存储系统 | HDFS/S3/GCS与训练集成 / 高性能并行文件系统（Lustre/GPFS）/ 显存与主存的数据流 / 存储带宽分析 |
| **Week 26** | 监控与可观测性 | DCGM（NVIDIA数据中心GPU管理）/ Prometheus+Grafana / 训练任务监控指标 / 告警与SLA |

---

## 阶段三：前沿深入（Week 27-40）

### 🔷 Module 7：注意力机制优化（Week 27-30）

| 周次 | 主题 | 关键内容 |
|------|------|---------|
| **Week 27** | FlashAttention 深度解析 | IO感知的注意力算法 / Tiling策略 / Online Softmax / FlashAttention-2/3改进点 |
| **Week 28** | 长序列训练技术 | Ring Attention / Sequence Parallelism / 稀疏注意力（Sparse Attention）/ Sliding Window Attention |
| **Week 29** | MLA / GQA / MQA | Multi-Query Attention / Grouped-Query Attention / KV Head压缩 / DeepSeek MLA实现 |
| **Week 30** | 位置编码与外推 | RoPE原理与实现 / ALiBi / NTK外推 / YaRN / 长文本外推评估 |

---

### 🔷 Module 8：MOE 与稀疏模型（Week 31-33）

| 周次 | 主题 | 关键内容 |
|------|------|---------|
| **Week 31** | MOE 原理与实现 | Mixture of Experts架构 / Top-K路由 / Expert并行 / 负载均衡损失 / 容量因子 |
| **Week 32** | MOE 系统工程 | Expert并行通信（AlltoAll）/ MOE与TP/DP的结合 / DeepSeek-V3 MOE实现解析 |
| **Week 33** | 稀疏化与剪枝 | 结构化剪枝 / 非结构化剪枝 / 权重稀疏化 / 稀疏矩阵乘（cusparse）|

---

### 🔷 Module 9：强化学习训练系统（Week 34-37）

| 周次 | 主题 | 关键内容 |
|------|------|---------|
| **Week 34** | RLHF 系统架构 | PPO算法原理 / Actor-Critic / Reference Model / Reward Model / 训练流程 |
| **Week 35** | RLHF 工程实现 | TRL/OpenRLHF框架 / 多模型协同训练 / 显存估算与配置 / vLLM集成加速推理端 |
| **Week 36** | GRPO 与新型 RL 算法 | GRPO原理 / DPO/RAFT等无需RM的方法 / 训练稳定性 / DeepSeek-R1系统解析 |
| **Week 37** | 在线强化学习系统 | 异步RL架构 / 数据飞轮 / 经验回放 / 生产级RLHF系统设计 |

---

### 🔷 Module 10：多模态与生成模型基础设施（Week 38-40）

| 周次 | 主题 | 关键内容 |
|------|------|---------|
| **Week 38** | 多模态训练系统 | 视觉编码器（ViT）+ LLM的联合训练 / 模态对齐 / 数据混合策略 / LLaVA/Qwen-VL架构 |
| **Week 39** | 扩散模型基础设施 | Stable Diffusion训练/推理 / VAE/UNet加速 / xFormers集成 / DiT（Diffusion Transformer）|
| **Week 40** | 视频生成模型 | 时序建模 / 3D卷积与注意力 / 视频数据流水线 / 大规模视频训练工程挑战 |

---

## 阶段四：生产与前沿（Week 41-52）

### 🔷 Module 11：生产化与平台工程（Week 41-45）

| 周次 | 主题 | 关键内容 |
|------|------|---------|
| **Week 41** | 模型服务化 | 模型序列化（safetensors/GGUF）/ gRPC服务 / Triton Inference Server / 动态批处理 / 模型版本管理 |
| **Week 42** | 推理成本优化 | 显存优化策略全集 / 批处理策略 / 推理成本估算模型 / 实际生产案例分析 |
| **Week 43** | AI 平台工程 | 训练平台设计（数据/训练/评估一体化）/ 实验管理（MLflow/W&B）/ 模型 Registry / 持续训练流水线 |
| **Week 44** | 故障诊断与SRE | GPU 硬件故障检测 / NCCL超时排查 / 训练hang分析 / 大规模训练的SRE实践 |
| **Week 45** | 成本与效率工程 | GPU利用率优化 / 混合云调度 / Spot实例策略 / 训练效率度量（MFU/HFU）|

---

### 🔷 Module 12：前沿技术与综合项目（Week 46-52）

| 周次 | 主题 | 关键内容 |
|------|------|---------|
| **Week 46** | 新型硬件适配 | AMD ROCm生态 / TPU编程模型（JAX/XLA）/ 国产AI芯片（昇腾/寒武纪）适配思路 |
| **Week 47** | 编译器与内核生成 | Triton深度实战（自定义Attention Kernel）/ torch.compile源码解析 / TVM AutoTVM/Ansor |
| **Week 48** | 通信库源码解析 | NCCL源码：AllReduce实现 / 拓扑检测 / 树形vs环形算法选择 / 通信性能调优 |
| **Week 49** | 前沿论文精读 | 近期NeurIPS/MLSys论文精选（系统类）/ 原理 → 工程 → 实现的完整拆解 |
| **Week 50** | 综合项目：mini训练框架 | 从零实现支持TP+DP的Transformer训练框架 / 自定义算子 / 分布式通信 / Checkpoint |
| **Week 51** | 综合项目：推理服务 | 构建支持连续批处理的LLM推理服务 / PagedAttention手动实现 / 性能对比 |
| **Week 52** | 年度复盘与技术展望 | 52周知识体系梳理 / 个人技术树完善 / 行业动态与下一年规划 |

---

## 附录：学习资源推荐

### 书籍与教材
- 《Programming Massively Parallel Processors》（Kirk & Hwu）—— CUDA 圣经
- 《Computer Architecture: A Quantitative Approach》—— 体系结构基础
- 《Designing Data-Intensive Applications》—— 分布式系统思维

### 必读论文
- FlashAttention 1/2/3 系列
- Megatron-LM 系列（Tensor Parallelism, Sequence Parallelism）
- ZeRO / FSDP 系列
- vLLM (Efficient Memory Management for Large Language Model Serving)
- DeepSeek-V3 / DeepSeek-R1 技术报告

### 代码仓库
- [NVIDIA/apex](https://github.com/NVIDIA/apex) — 混合精度工具
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — 大模型训练框架
- [vllm-project/vllm](https://github.com/vllm-project/vllm) — 高效推理引擎
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — FlashAttention
- [openai/triton](https://github.com/openai/triton) — Triton 编译器

### 工具清单
- Nsight Systems / Nsight Compute（性能分析）
- Weights & Biases（实验追踪）
- DCGM（GPU集群监控）
- Netron（模型结构可视化）
- torch.profiler + TensorBoard

---

> 📌 **学习原则**：每周完成理论 + 动手实验 + 一道编程题。遇到不懂先跳过，一周后回头看往往豁然开朗。坚持比完美更重要。
>
> 🗓️ 更新频率：每完成一个 Module 后回顾更新，根据实际进度动态调整后续规划。
