# D1：GPU vs CPU——为什么 GPU 能加速深度学习？

> Week1 主题：GPU架构与CUDA基础 | AI Infra线

---

## 🎯 学习目标

- 理解 CPU 与 GPU 的核心设计哲学差异
- 掌握吞吐量导向 vs 延迟导向的计算模型
- 能用自己的语言解释"为什么深度学习需要 GPU"
- 了解现代 GPU（A100/H100）的基本参数量级

---

## 🧠 核心知识点

### 1. CPU 的设计哲学：降低延迟

CPU 的设计目标是让**单个任务尽快完成**：
- 少量但强大的核心（现代服务器 CPU：8~96 核）
- 大容量多级缓存（L1/L2/L3，总计几十 MB）
- 乱序执行（Out-of-Order Execution）、分支预测（Branch Prediction）
- 超线程（Hyperthreading）

CPU 擅长：复杂控制流、低延迟单线程任务、操作系统调度

### 2. GPU 的设计哲学：最大化吞吐量

GPU 的设计目标是让**大量相似任务并发完成**：
- 海量但简单的核心（H100：16896 个 CUDA 核心）
- 小缓存（寄存器文件极大，但共享内存有限）
- 简单的控制流（SIMD/SIMT 执行模式）
- 隐藏内存延迟的关键：**线程级并行（TLP）**

GPU 擅长：数据并行计算、矩阵运算、大批量独立任务

### 3. 核心数量与内存带宽对比

| 指标 | AMD EPYC 9654（CPU） | NVIDIA H100 SXM（GPU） |
|------|---------------------|----------------------|
| 计算单元数 | 96 核 | 16896 CUDA 核心 |
| FP32 算力 | ~7 TFLOPS | 67 TFLOPS (TF32: 989 TOPS) |
| 内存带宽 | ~460 GB/s | 3.35 TB/s |
| 内存类型 | DDR5 | HBM3 |
| TDP | 360W | 700W |

### 4. 内存层级对比

```
CPU 内存层级：
  寄存器 (< 1ns)
    → L1 Cache ~32KB/core (~1ns)
      → L2 Cache ~1MB/core (~5ns)
        → L3 Cache ~共享几十MB (~20ns)
          → DRAM 主存 (~100ns, ~50 GB/s)

GPU 内存层级：
  寄存器 (每个线程独有, 极快)
    → L1 Cache / 共享内存 (~32KB~164KB/SM)
      → L2 Cache (~40~50MB on H100)
        → HBM 显存 (~3.35 TB/s, ~1GB延迟几百ns)
          → PCIe → CPU主存（瓶颈！）
```

### 5. 深度学习为什么天然适合 GPU？

神经网络前向传播的核心：**矩阵乘法（GEMM）**

```
Y = X @ W + b

X: [batch_size, input_dim]
W: [input_dim, output_dim]
```

这个操作有以下特性：
- **高度数据并行**：输出矩阵的每个元素都可独立计算
- **算术密集**：大量乘加操作（MACs），计算密集型
- **规则内存访问**：便于向量化和缓存预取

GPU 的 SIMT（Single Instruction Multiple Threads）模型完美匹配这种工作负载。

### 6. Roofline 模型：计算瓶颈分析

```
算术强度（Arithmetic Intensity）= FLOPs / 字节数（内存访问）

内存带宽上限（Memory Bound）: 性能 = 带宽 × 算术强度
算力上限（Compute Bound）:    性能 = 峰值算力

Ridge Point = 峰值算力 / 峰值带宽
H100: 67 TFLOPS / 3.35 TB/s ≈ 20 FLOPs/Byte
```

深度学习算子：
- **矩阵乘（大 batch）**：算术强度高 → Compute Bound（GPU 优势区）
- **Elementwise 激活函数**：算术强度低 → Memory Bound
- **小 batch 推理**：往往 Memory Bound

---

## 🖼️ 原理图解

### CPU vs GPU 架构示意

```
┌─────────────────────────────────────────────────────────┐
│                         CPU                              │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                   │
│  │ Core │ │ Core │ │ Core │ │ Core │  ← 复杂核心         │
│  │ ALU  │ │ ALU  │ │ ALU  │ │ ALU  │    分支预测         │
│  │ FPU  │ │ FPU  │ │ FPU  │ │ FPU  │    乱序执行         │
│  └──────┘ └──────┘ └──────┘ └──────┘                   │
│  ┌────────────────────────────────────────────────────┐  │
│  │              大型 L3 Cache (共享)                   │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                         GPU                              │
│  ┌──────────────────────────────────────────────────┐   │
│  │  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  │   │
│  │  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  │   │
│  │  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  │   │
│  │  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  SM  │   │
│  │       ↑ 每个SM含128个CUDA核心（H100）              │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │                L2 Cache (50MB)                    │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │              HBM3 显存 (80GB, 3.35TB/s)           │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Roofline 图示

```
性能 ^
(GFLOPS) │            _____________________ 峰值算力线
         │           /
         │          /  Compute Bound 区域
         │         /
         │        / ← Ridge Point
         │  ___  /
         │ /   \/
         │/  Memory Bound 区域
         └─────────────────────────────────→
                                     算术强度 (FLOPs/Byte)
```

---

## 🛠️ 动手练习

### 练习 1：GPU 规格查询与计算

查询以下 GPU 的峰值 FP32 算力和内存带宽，并计算 Ridge Point：

| GPU | FP32 算力 | 内存带宽 | Ridge Point |
|-----|----------|---------|-------------|
| RTX 3090 | ? | ? | ? |
| A100 SXM | ? | ? | ? |
| H100 SXM | ? | ? | ? |

> 参考：NVIDIA官网规格表、[techpowerup GPU DB](https://www.techpowerup.com/gpu-specs/)

### 练习 2：矩阵乘法算术强度计算

计算以下矩阵乘法的算术强度（假设 FP32，即每个元素 4 字节）：

```
A: [1024, 1024]  ×  B: [1024, 1024]  →  C: [1024, 1024]

FLOPs = ?      （每次输出元素需要 1024 次乘法 + 1024 次加法）
内存访问 = ?    （读 A + 读 B + 写 C）
算术强度 = ?
```

### 练习 3：环境准备

```bash
# 检查是否有可用 GPU
nvidia-smi

# 查看 CUDA 版本
nvcc --version

# Python 中检查 GPU
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_properties(0))"

# 安装 PyTorch（如未安装）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 练习 4：CPU vs GPU 速度对比（Python）

```python
import torch
import time

# 矩阵大小
N = 4096

# CPU 矩阵乘法
A_cpu = torch.randn(N, N)
B_cpu = torch.randn(N, N)

start = time.time()
for _ in range(10):
    C_cpu = A_cpu @ B_cpu
cpu_time = (time.time() - start) / 10
print(f"CPU 平均耗时: {cpu_time*1000:.2f} ms")

# GPU 矩阵乘法（如果可用）
if torch.cuda.is_available():
    A_gpu = A_cpu.cuda()
    B_gpu = B_cpu.cuda()
    
    # Warmup
    for _ in range(3):
        C_gpu = A_gpu @ B_gpu
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(10):
        C_gpu = A_gpu @ B_gpu
    torch.cuda.synchronize()
    gpu_time = (time.time() - start) / 10
    print(f"GPU 平均耗时: {gpu_time*1000:.2f} ms")
    print(f"加速比: {cpu_time/gpu_time:.1f}x")
```

> 记录你的实验结果，思考：为什么小矩阵时 GPU 优势不明显？

---

## 📝 小结

| 维度 | CPU | GPU |
|------|-----|-----|
| 设计目标 | 低延迟，通用计算 | 高吞吐，数据并行 |
| 核心数量 | 少（8~96） | 多（几千~万级） |
| 内存带宽 | 50~460 GB/s | 1~3.35 TB/s |
| 擅长任务 | 复杂控制流、串行逻辑 | 矩阵运算、大批量并行 |
| 深度学习角色 | 数据预处理、调度 | 训练/推理的核心算力 |

**关键认知**：GPU 不是更快的 CPU，而是**完全不同设计哲学**的处理器。它以牺牲单线程性能和通用性，换取极高的并行吞吐量——这正是深度学习工作负载的需求所在。

---

**明日预告**：D2 将深入 GPU 内部——SM（Streaming Multiprocessor）架构详解，理解 Warp、Thread Block 的执行原理。
