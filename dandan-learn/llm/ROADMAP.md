# 🧠 大模型学习线 — 52周详细规划

> **项目**：丹丹的 AI + 大模型 + 量化一年速成计划  
> **目标**：从零基础到顶尖水平，系统掌握大模型原理、工程实践与前沿研究  
> **节奏**：每周5天学习（周六日休息或复习），每天1篇精讲 + 代码实践  
> **输出**：每日 `dayXXX.md` 保存到对应 `weekXX/` 目录

---

## 📌 总体阶段划分

| 阶段 | 周次 | 主题 | 关键词 |
|------|------|------|--------|
| **Phase 1** | Week 01–08 | 基础夯实 | Python/数学/NLP入门/Transformer原理 |
| **Phase 2** | Week 09–20 | 核心技术 | 预训练/微调/RLHF/推理优化 |
| **Phase 3** | Week 21–36 | 工程实战 | RAG/Agent/部署/评测/安全 |
| **Phase 4** | Week 37–48 | 前沿研究 | 多模态/代码大模型/长文本/科学AI |
| **Phase 5** | Week 49–52 | 综合提升 | 论文精读/毕业项目/职业规划 |

---

## 🔵 Phase 1：基础夯实（Week 01–08）

### Week 01 — Python & 工具链
| 天 | 主题 | 文件 |
|----|------|------|
| Day 1 (周一) | Python环境搭建 + Jupyter/VS Code配置 | week01/day001.md |
| Day 2 (周二) | Python核心语法：列表、字典、函数、类 | week01/day002.md |
| Day 3 (周三) | NumPy + Pandas 数据处理基础 | week01/day003.md |
| Day 4 (周四) | Matplotlib 数据可视化 | week01/day004.md |
| Day 5 (周五) | Git版本控制 + GitHub工作流 | week01/day005.md |

### Week 02 — 数学基础 I：线性代数
| 天 | 主题 | 文件 |
|----|------|------|
| Day 6 | 向量与矩阵：从直觉到计算 | week02/day006.md |
| Day 7 | 矩阵乘法 + 转置 + 行列式 | week02/day007.md |
| Day 8 | 特征值与特征向量（PCA铺垫） | week02/day008.md |
| Day 9 | SVD奇异值分解 | week02/day009.md |
| Day 10 | 用NumPy实现线性代数操作 | week02/day010.md |

### Week 03 — 数学基础 II：微积分 + 概率
| 天 | 主题 | 文件 |
|----|------|------|
| Day 11 | 导数与梯度直觉理解 | week03/day011.md |
| Day 12 | 链式法则 + 反向传播推导 | week03/day012.md |
| Day 13 | 概率论基础：贝叶斯定理、条件概率 | week03/day013.md |
| Day 14 | 信息论：熵、交叉熵、KL散度 | week03/day014.md |
| Day 15 | 统计学习基础：极大似然估计 | week03/day015.md |

### Week 04 — 深度学习基础 I
| 天 | 主题 | 文件 |
|----|------|------|
| Day 16 | 神经网络结构：感知机→多层感知机 | week04/day016.md |
| Day 17 | 激活函数：ReLU/Sigmoid/GELU | week04/day017.md |
| Day 18 | 损失函数 + 梯度下降 | week04/day018.md |
| Day 19 | PyTorch入门：Tensor + Autograd | week04/day019.md |
| Day 20 | 用PyTorch搭建第一个MLP | week04/day020.md |

### Week 05 — 深度学习基础 II
| 天 | 主题 | 文件 |
|----|------|------|
| Day 21 | CNN卷积神经网络原理 | week05/day021.md |
| Day 22 | BatchNorm + Dropout + 正则化 | week05/day022.md |
| Day 23 | RNN/LSTM/GRU序列建模 | week05/day023.md |
| Day 24 | Attention机制起源（Bahdanau Attention） | week05/day024.md |
| Day 25 | 训练技巧：学习率调度、早停、权重初始化 | week05/day025.md |

### Week 06 — NLP基础
| 天 | 主题 | 文件 |
|----|------|------|
| Day 26 | 文本预处理：分词、词汇表、one-hot | week06/day026.md |
| Day 27 | Word2Vec：Skip-gram & CBOW | week06/day027.md |
| Day 28 | GloVe + FastText 词向量 | week06/day028.md |
| Day 29 | 语言模型基础：n-gram → 神经语言模型 | week06/day029.md |
| Day 30 | Subword分词：BPE / WordPiece / SentencePiece | week06/day030.md |

### Week 07 — Transformer 原理精讲
| 天 | 主题 | 文件 |
|----|------|------|
| Day 31 | Self-Attention机制推导（Q/K/V） | week07/day031.md |
| Day 32 | Multi-Head Attention + 位置编码 | week07/day032.md |
| Day 33 | Transformer Encoder详解 | week07/day033.md |
| Day 34 | Transformer Decoder + 掩码机制 | week07/day034.md |
| Day 35 | 手写Transformer（PyTorch实现） | week07/day035.md |

### Week 08 — BERT & GPT 里程碑
| 天 | 主题 | 文件 |
|----|------|------|
| Day 36 | BERT：预训练任务MLM + NSP | week08/day036.md |
| Day 37 | BERT微调：文本分类、NER、QA | week08/day037.md |
| Day 38 | GPT系列：GPT-1/2/3演进 | week08/day038.md |
| Day 39 | 自回归 vs 自编码：两种范式对比 | week08/day039.md |
| Day 40 | Phase 1 总结 + 小测验 | week08/day040.md |

---

## 🟡 Phase 2：核心技术（Week 09–20）

### Week 09 — 大规模预训练
| 天 | 主题 | 文件 |
|----|------|------|
| Day 41 | Scaling Law：规模定律与涌现能力 | week09/day041.md |
| Day 42 | 数据工程：预训练数据收集与清洗 | week09/day042.md |
| Day 43 | 分布式训练基础：数据并行 vs 模型并行 | week09/day043.md |
| Day 44 | Megatron-LM / DeepSpeed 框架 | week09/day044.md |
| Day 45 | 混合精度训练（FP16/BF16） | week09/day045.md |

### Week 10 — 现代大模型架构
| 天 | 主题 | 文件 |
|----|------|------|
| Day 46 | LLaMA系列架构：RoPE + RMSNorm + SwiGLU | week10/day046.md |
| Day 47 | GPT-NeoX / Mistral / Mixtral解析 | week10/day047.md |
| Day 48 | MoE混合专家模型 | week10/day048.md |
| Day 49 | Flash Attention：IO感知注意力优化 | week10/day049.md |
| Day 50 | KV Cache与推理加速 | week10/day050.md |

### Week 11 — Prompt Engineering
| 天 | 主题 | 文件 |
|----|------|------|
| Day 51 | Prompt基础：零样本/少样本学习 | week11/day051.md |
| Day 52 | Chain-of-Thought（CoT）思维链 | week11/day052.md |
| Day 53 | Tree of Thoughts / Graph of Thoughts | week11/day053.md |
| Day 54 | Prompt安全：越狱攻防与防御策略 | week11/day054.md |
| Day 55 | 系统级Prompt设计实战 | week11/day055.md |

### Week 12 — 指令微调（SFT）
| 天 | 主题 | 文件 |
|----|------|------|
| Day 56 | SFT原理：指令跟随数据集构建 | week12/day056.md |
| Day 57 | LoRA：低秩自适应微调 | week12/day057.md |
| Day 58 | QLoRA：4bit量化+LoRA | week12/day058.md |
| Day 59 | LLaMA-Factory / Axolotl 实战微调 | week12/day059.md |
| Day 60 | 微调踩坑：过拟合/灾难遗忘/数据质量 | week12/day060.md |

### Week 13 — RLHF 与对齐
| 天 | 主题 | 文件 |
|----|------|------|
| Day 61 | RLHF原理：奖励模型 + PPO | week13/day061.md |
| Day 62 | DPO：直接偏好优化 | week13/day062.md |
| Day 63 | Constitutional AI（Anthropic方法） | week13/day063.md |
| Day 64 | 对齐税：能力与安全的权衡 | week13/day064.md |
| Day 65 | 实战：用TRL库做简单RLHF | week13/day065.md |

### Week 14 — 推理优化 I
| 天 | 主题 | 文件 |
|----|------|------|
| Day 66 | 量化基础：INT8 / INT4 / GPTQ | week14/day066.md |
| Day 67 | AWQ / SmoothQuant 量化方案 | week14/day067.md |
| Day 68 | 剪枝：结构化 vs 非结构化 | week14/day068.md |
| Day 69 | 知识蒸馏：从大模型到小模型 | week14/day069.md |
| Day 70 | vLLM推理框架：PagedAttention | week14/day070.md |

### Week 15 — 推理优化 II
| 天 | 主题 | 文件 |
|----|------|------|
| Day 71 | Speculative Decoding投机解码 | week15/day071.md |
| Day 72 | Continuous Batching 连续批处理 | week15/day072.md |
| Day 73 | Tensor并行推理：TensorRT-LLM | week15/day073.md |
| Day 74 | 边缘设备部署：llama.cpp / MLC-LLM | week15/day074.md |
| Day 75 | 推理Benchmark与性能调优 | week15/day075.md |

### Week 16 — 评测体系
| 天 | 主题 | 文件 |
|----|------|------|
| Day 76 | LLM评测维度：能力/安全/效率 | week16/day076.md |
| Day 77 | 标准Benchmark：MMLU/HellaSwag/HumanEval | week16/day077.md |
| Day 78 | 中文评测：C-Eval / CMMLU / SuperCLUE | week16/day078.md |
| Day 79 | 自动评测 vs 人工评测 | week16/day079.md |
| Day 80 | 实战：用lm-evaluation-harness跑评测 | week16/day080.md |

### Week 17 — 数据飞轮
| 天 | 主题 | 文件 |
|----|------|------|
| Day 81 | 数据质量 vs 数量：质量优先原则 | week17/day081.md |
| Day 82 | 合成数据：Self-Instruct / Alpaca方法 | week17/day082.md |
| Day 83 | 数据去重：MinHash / SimHash | week17/day083.md |
| Day 84 | 数据标注工程：Label Studio实战 | week17/day084.md |
| Day 85 | 数据版权与合规 | week17/day085.md |

### Week 18 — 开源生态全景
| 天 | 主题 | 文件 |
|----|------|------|
| Day 86 | Hugging Face生态：Transformers/Hub/Spaces | week18/day086.md |
| Day 87 | LangChain框架深度解析 | week18/day087.md |
| Day 88 | LlamaIndex：结构化知识检索 | week18/day088.md |
| Day 89 | Ollama：本地大模型一键运行 | week18/day089.md |
| Day 90 | OpenAI API + 主流API对比 | week18/day090.md |

### Week 19 — 安全与对齐深化
| 天 | 主题 | 文件 |
|----|------|------|
| Day 91 | 幻觉问题：成因分析与缓解策略 | week19/day091.md |
| Day 92 | Prompt注入攻击与防御 | week19/day092.md |
| Day 93 | 偏见与公平性 | week19/day093.md |
| Day 94 | 版权与水印技术 | week19/day094.md |
| Day 95 | AI治理：政策法规与伦理框架 | week19/day095.md |

### Week 20 — Phase 2 综合复习
| 天 | 主题 | 文件 |
|----|------|------|
| Day 96 | 核心技术体系梳理 | week20/day096.md |
| Day 97 | 实战项目：从零微调一个垂直领域模型 | week20/day097.md |
| Day 98 | 实战项目：量化+部署本地推理服务 | week20/day098.md |
| Day 99 | 论文阅读方法论：如何高效读AI论文 | week20/day099.md |
| Day 100 | 100天里程碑 🎉 总结与展望 | week20/day100.md |

---

## 🟠 Phase 3：工程实战（Week 21–36）

### Week 21 — RAG系统 I：基础架构
| 天 | 主题 | 文件 |
|----|------|------|
| Day 101 | RAG原理：检索增强生成全景图 | week21/day101.md |
| Day 102 | 向量数据库：Faiss / Chroma / Milvus | week21/day102.md |
| Day 103 | Embedding模型选型与评测 | week21/day103.md |
| Day 104 | 文档解析：PDF/Word/网页处理 | week21/day104.md |
| Day 105 | 检索策略：稠密检索 + 稀疏检索 + 混合检索 | week21/day105.md |

### Week 22 — RAG系统 II：进阶优化
| 天 | 主题 | 文件 |
|----|------|------|
| Day 106 | Re-ranking重排序：Cohere/BGE | week22/day106.md |
| Day 107 | HyDE假设文档嵌入 | week22/day107.md |
| Day 108 | 多跳推理 RAG | week22/day108.md |
| Day 109 | 知识图谱 + RAG（GraphRAG） | week22/day109.md |
| Day 110 | RAG评测：RAGAS框架 | week22/day110.md |

### Week 23 — Agent系统 I：基础
| 天 | 主题 | 文件 |
|----|------|------|
| Day 111 | Agent概念：思维-行动-观察循环 | week23/day111.md |
| Day 112 | ReAct框架实现 | week23/day112.md |
| Day 113 | 工具调用（Function Calling）实战 | week23/day113.md |
| Day 114 | 代码执行Agent：代码解释器 | week23/day114.md |
| Day 115 | 记忆系统：短期记忆 vs 长期记忆 | week23/day115.md |

### Week 24 — Agent系统 II：多智能体
| 天 | 主题 | 文件 |
|----|------|------|
| Day 116 | Multi-Agent框架：AutoGen / CrewAI | week24/day116.md |
| Day 117 | 任务规划与分解：CAMEL / BabyAGI | week24/day117.md |
| Day 118 | Agent评测与可靠性 | week24/day118.md |
| Day 119 | 生产级Agent：错误处理与容错 | week24/day119.md |
| Day 120 | 实战：搭建一个自主研究Agent | week24/day120.md |

### Week 25 — 生产部署 I
| 天 | 主题 | 文件 |
|----|------|------|
| Day 121 | API服务设计：FastAPI + 流式输出 | week25/day121.md |
| Day 122 | Docker容器化大模型服务 | week25/day122.md |
| Day 123 | Kubernetes部署 + 弹性扩缩容 | week25/day123.md |
| Day 124 | 负载均衡 + 并发压测 | week25/day124.md |
| Day 125 | 监控与可观测性：日志/指标/追踪 | week25/day125.md |

### Week 26 — 生产部署 II
| 天 | 主题 | 文件 |
|----|------|------|
| Day 126 | GPU集群管理：SLURM / Ray | week26/day126.md |
| Day 127 | 多模型路由与网关 | week26/day127.md |
| Day 128 | 成本优化：请求缓存 / 批量推理 | week26/day128.md |
| Day 129 | A/B测试与灰度发布 | week26/day129.md |
| Day 130 | SLA保障与故障排查 | week26/day130.md |

### Week 27–28 — 垂直行业实战 I（金融/法律）
| 天 | 主题 | 文件 |
|----|------|------|
| Day 131 | 金融NLP：情感分析、事件抽取 | week27/day131.md |
| Day 132 | 量化辅助：用LLM分析财报 | week27/day132.md |
| Day 133 | 法律AI：合同审查、法规问答 | week27/day133.md |
| Day 134 | 医疗AI：诊断辅助、医学文献 | week27/day134.md |
| Day 135 | 垂直领域数据构建方法论 | week27/day135.md |
| Day 136-140 | 行业项目实战（自选） | week28/ |

### Week 29–30 — 垂直行业实战 II（代码/教育）
| 天 | 主题 | 文件 |
|----|------|------|
| Day 141 | 代码生成：GitHub Copilot原理 | week29/day141.md |
| Day 142 | 代码补全 vs 代码生成 vs 代码审查 | week29/day142.md |
| Day 143 | 教育AI：个性化学习路径 | week29/day143.md |
| Day 144 | 对话系统：客服机器人工程实践 | week29/day144.md |
| Day 145 | 内容生成：SEO+创意写作 | week29/day145.md |
| Day 146-150 | 行业项目实战（自选） | week30/ |

### Week 31–32 — LLMOps
| 天 | 主题 | 文件 |
|----|------|------|
| Day 151 | LLMOps概念与工具链 | week31/day151.md |
| Day 152 | Prompt版本管理 + 实验追踪 | week31/day152.md |
| Day 153 | 持续微调 + 数据版本控制 | week31/day153.md |
| Day 154 | 模型注册 + 上线流程 | week31/day154.md |
| Day 155 | 线上监控：漂移检测/质量告警 | week31/day155.md |
| Day 156-160 | LLMOps综合项目 | week32/ |

### Week 33–34 — 应用开发全栈
| 天 | 主题 | 文件 |
|----|------|------|
| Day 161 | 前端接入：React + Streaming UI | week33/day161.md |
| Day 162 | Vercel AI SDK / Next.js实战 | week33/day162.md |
| Day 163 | 移动端集成：iOS/Android LLM接入 | week33/day163.md |
| Day 164 | 插件系统：ChatGPT Plugin / MCP协议 | week33/day164.md |
| Day 165 | 产品化思维：从Demo到产品 | week33/day165.md |
| Day 166-170 | 全栈应用实战项目 | week34/ |

### Week 35–36 — Phase 3 综合项目
- 自选一个真实场景（推荐：量化研究助手 / 智能知识库 / 代码助手）
- 完整走通：需求→数据→微调→部署→评测→上线
- 撰写项目文档与技术报告

---

## 🔴 Phase 4：前沿研究（Week 37–48）

### Week 37–38 — 多模态大模型
| 主题 | 文件 |
|------|------|
| CLIP / BLIP / Flamingo视觉-语言模型 | week37/ |
| GPT-4V / Gemini Vision原理 | week37/ |
| 文生图：Stable Diffusion / DALL-E 3 | week38/ |
| 视频理解：VideoLLaMA / InternVideo | week38/ |
| 多模态Agent实战 | week38/ |

### Week 39–40 — 长文本与记忆
| 主题 | 文件 |
|------|------|
| 长上下文挑战：Lost in the Middle | week39/ |
| RoPE扩展：YaRN / LongRoPE | week39/ |
| Mamba / S4状态空间模型 | week40/ |
| 外部记忆：MemGPT | week40/ |
| 实战：100K上下文文档问答 | week40/ |

### Week 41–42 — 代码大模型
| 主题 | 文件 |
|------|------|
| CodeLLaMA / DeepSeek-Coder解析 | week41/ |
| 代码执行与单元测试生成 | week41/ |
| SWE-bench：软件工程Benchmark | week42/ |
| 自动化编程Agent：Devin原理 | week42/ |
| 量化策略代码生成实战 | week42/ |

### Week 43–44 — 推理与规划能力
| 主题 | 文件 |
|------|------|
| o1/o3思维链：慢思考机制 | week43/ |
| MCTS蒙特卡洛树搜索 + LLM | week43/ |
| 数学推理：GSM8K / MATH | week44/ |
| 形式化验证与LLM | week44/ |
| 推理增强微调 | week44/ |

### Week 45–46 — 科学AI
| 主题 | 文件 |
|------|------|
| AlphaFold蛋白质结构预测 | week45/ |
| 材料科学与分子生成 | week45/ |
| 气候模型与科学计算 | week46/ |
| BioMedLM / Med-PaLM | week46/ |

### Week 47–48 — 前沿架构探索
| 主题 | 文件 |
|------|------|
| Test-time Compute推理时计算 | week47/ |
| World Model世界模型 | week47/ |
| LLM + 强化学习：GRPO / PPO对比 | week48/ |
| 神经符号融合 | week48/ |

---

## 🟣 Phase 5：综合提升（Week 49–52）

### Week 49 — 顶会论文精读
- NeurIPS / ICML / ICLR 精选10篇深度解读
- 论文复现实践

### Week 50 — 毕业项目
- 自选综合项目（含量化线整合）
- 完整技术文档 + 演示Demo

### Week 51 — 职业规划
- 大模型岗位图谱：研究员/工程师/产品经理
- 简历优化 + 面试准备
- 开源贡献指南

### Week 52 — 总结与展望
- 一年学习总结
- 技术雷达：2027年大模型预判
- 下一步：读研/创业/就业路径

---

## 📁 目录结构说明

```
dandan-learn/llm/
├── ROADMAP.md          # 本文件：52周规划总览
├── week01/             # 每周对应目录
│   ├── day001.md       # 每日学习内容
│   ├── day002.md
│   └── ...
├── week02/
│   └── ...
...
└── week52/
```

---

## 💡 学习建议

1. **每天30-60分钟**：碎片时间读理论，整块时间做实践
2. **代码优先**：每篇 `dayXXX.md` 都附带可运行代码片段
3. **知识管理**：每周五做周总结，每月做月报
4. **社区参与**：用学到的知识在群里讨论，教是最好的学
5. **量化联动**：大模型线的金融/量化内容主动与量化线同学交流

---

*最后更新：2026-04-14 | 版本：v1.0*
