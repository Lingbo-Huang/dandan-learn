# 大模型算法工程师学习路线图

本文档为从Java工程师转型为大模型算法工程师的学习路线图，涵盖了从数学基础到具体算法实现的完整学习路径。

## 一、学习路线总览

1. **数学基础**（建议时间：2-3个月）
   - 线性代数（向量、矩阵等）
   - 微积分
   - 概率论与统计
2. **机器学习基础**（建议时间：1-2个月）
   - 监督学习、无监督学习、强化学习基础概念
   - 常见机器学习算法（线性回归、逻辑回归、决策树等）
3. **深度学习基础**（建议时间：2-3个月）
   - 神经网络基础
   - CNN、RNN、Transformer架构
4. **大模型核心技术**（建议时间：3-4个月）
   - 预训练 (PT)
   - 监督微调 (SFT)
   - 偏好对齐算法（DPO、PPO、GRPO、RLAIF、ORPO、SimPO、KTO等）
5. **实践项目**（建议时间：2-3个月）
   - 复现经典算法
   - 参与开源项目
   - 构建自己的应用

## 二、阶段一：数学基础

### 2.1 学习内容

#### 线性代数
- 向量及其运算（加法、数乘、点积、叉积）
- 矩阵及其运算（加法、乘法、转置、逆）
- 行列式
- 特征值与特征向量
- 奇异值分解 (SVD)
- 二次型

#### 微积分
- 导数与偏导数
- 梯度、散度、旋度
- 链式法则
- 泰勒展开
- 积分基础

#### 概率论与统计
- 随机事件与概率
- 条件概率与贝叶斯公式
- 随机变量（离散型、连续型）
- 概率分布（正态分布、伯努利分布等）
- 数学期望与方差
- 协方差与相关系数
- 参数估计（最大似然估计）
- 信息论基础（熵、KL散度）

### 2.2 验证手段

- 完成相关课程的作业和考试（如MIT的线性代数课程、可汗学院的概率统计课程）
- 使用Python/Numpy手动实现基础运算（如矩阵乘法、求特征值等）
- 阅读并理解机器学习经典论文中的数学推导部分

## 三、阶段二：PT、SFT、RL关键技术

### 3.1 预训练 (Pre-Training, PT)

#### 学习内容
- 自回归模型与自编码模型
- Transformer架构详解（自注意力机制、位置编码等）
- 预训练任务（如MLM、NSP、因果语言建模）
- 数据预处理与Tokenization

#### 验证手段
- 阅读BERT、GPT系列论文
- 使用HuggingFace Transformers库加载并运行预训练模型
- 尝试修改模型输入，观察输出变化

### 3.2 监督微调 (Supervised Fine-Tuning, SFT)

#### 学习内容
- SFT的基本概念与流程
- 全量参数微调 (FFT) 与参数高效微调 (PEFT)
   - LoRA (Low-Rank Adaptation)
   - QLoRA (Quantized LoRA)
- 指令微调 (Instruction Tuning)
- 数据构建与清洗

#### 验证手段
- 在公开数据集（如GLUE、SuperGLUE）上微调模型并评估性能
- 实现一个简单的LoRA微调脚本
- 对比SFT前后模型在特定任务上的表现差异

### 3.3 偏好对齐算法

#### 学习内容

##### RLHF (Reinforcement Learning from Human Feedback)
- 奖励模型 (Reward Model, RM) 的训练
- PPO (Proximal Policy Optimization) 算法原理与实现
- RLHF的整体流程与挑战

##### RLAIF (Reinforcement Learning from AI Feedback)
- RLAIF与RLHF的区别
- AI生成偏好数据的优势与挑战
- RLAIF的应用场景

##### DPO (Direct Preference Optimization)
- DPO算法原理（基于Bradley-Terry模型）
- 偏好数据的构造
- DPO损失函数推导与实现
- DPO与PPO的对比分析

##### GRPO (Group Relative Policy Optimization)
- GRPO算法原理（群体对比、无需价值函数）
- GRPO在长序列任务中的优势
- GRPO与PPO的对比分析

##### ORPO (Odds Ratio Preference Optimization)
- ORPO算法原理（结合SFT和偏好优化）
- 无需参考模型的优势
- ORPO损失函数解析

##### SimPO (Simple Preference Optimization)
- SimPO算法原理（简化DPO，无需参考模型和奖励函数）
- 基于平均对数概率的优化目标
- SimPO与DPO的对比

##### KTO (Kahneman-Tversky Optimization)
- KTO算法原理（基于前景理论的优化）
- 单边优化策略
- KTO与DPO的对比

#### 验证手段
- 阅读PPO、DPO、GRPO、ORPO、SimPO、KTO原始论文
- 使用模拟数据手动实现各种算法的损失函数计算
- 在简单环境（如文本摘要、对话生成）中尝试应用不同算法进行微调
- 分析不同算法在相同数据下的效果差异

## 四、学习资源推荐

### 数学基础
- **书籍**：
  - 《线性代数及其应用》（David C. Lay）
  - 《托马斯微积分》
  - 《概率论与数理统计》（盛骤等）
- **在线课程**：
  - MIT 18.06 Linear Algebra (Gilbert Strang)
  - Khan Academy Probability and Statistics

### 机器学习与深度学习
- **书籍**：
  - 《机器学习》（周志华）
  - 《深度学习》（Ian Goodfellow等）
- **在线课程**：
  - Andrew Ng的Machine Learning Course (Coursera)
  - Deep Learning Specialization (Coursera)

### 大模型技术
- **论文**：
  - Attention Is All You Need (Transformer)
  - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
  - Training language models to follow instructions with human feedback (InstructGPT)
  - Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)
  - KTO: Model Alignment as Prospect Theoretic Optimization
  - ORPO: Monolithic Preference Optimization without Reference Model
  - SimPO: Simple Preference Optimization with a Reference-Free Reward
- **代码库**：
  - HuggingFace Transformers
  - Transformers Library from HuggingFace
  - LLaMA & Alpaca
- **博客与教程**：
  - HuggingFace Blog
  - Lil'Log (https://lilianweng.github.io/)