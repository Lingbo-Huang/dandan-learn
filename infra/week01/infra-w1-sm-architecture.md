# D2：SM 架构详解——GPU 的计算心脏

> Week1 主题：GPU架构与CUDA基础 | AI Infra线

---

## 🎯 学习目标

- 理解 SM（Streaming Multiprocessor）的内部组成结构
- 掌握 Warp、Thread、Block、Grid 的概念及层次关系
- 理解 SIMT 执行模型与 Warp Divergence
- 了解 SM 资源（寄存器、共享内存）对 Occupancy 的影响

---

## 🧠 核心知识点

### 1. GPU 的整体层级结构

```
GPU
├── GPC（Graphics Processing Cluster，图形处理器集群）
│   ├── TPC（Texture Processing Cluster）
│   │   ├── SM（Streaming Multiprocessor）← 核心计算单元
│   │   └── SM
│   └── TPC
└── ...（多个GPC）

内存层级（独立于上述计算层级）：
├── 寄存器文件（Register File，每个线程独有）
├── L1 Cache + 共享内存（每个SM独有，可配置比例）
├── L2 Cache（芯片级共享）
└── HBM 全局显存（所有SM共享）
```

**H100 SXM5 规格**：
- 132 个 SM
- 每个 SM：128 个 CUDA FP32 核心
- 总计：16896 个 CUDA 核心
- 每个 SM：256KB 寄存器文件，228KB L1/共享内存

### 2. SM 内部结构（以 Ampere 架构为例）

一个 SM 包含（A100）：
- **4 个 Warp Scheduler**（每个 SM 可同时调度 4 个 Warp）
- **4 个 Dispatch Unit**（指令分发器）
- **128 个 FP32 CUDA Core**（4组×32个）
- **64 个 INT32 Core**
- **4 个第三代 Tensor Core**（执行矩阵运算 MMA）
- **256KB 寄存器文件**（所有驻留线程共享）
- **192KB L1/共享内存**（可配置为 0/8/16/32/64/100/132/164KB 共享内存）
- **L1 指令缓存**

### 3. 线程层级（Thread Hierarchy）

CUDA 的执行层级：

```
Grid（网格）         ← 整个 kernel 的执行范围
  └── Block（线程块） ← 在一个 SM 上执行，线程间可通过共享内存通信
        └── Warp     ← 32个线程，SM 调度的最小单位（硬件概念）
              └── Thread（线程） ← 最小执行单元
```

**关键规则**：
- 一个 Block 必须在同一个 SM 上执行（不可跨 SM）
- 一个 SM 可以同时驻留多个 Block（资源允许时）
- Warp = 32 个连续线程（Thread 0~31 为 Warp 0，以此类推）
- 一个 Block 内的线程可以通过 `__syncthreads()` 同步
- 不同 Block 间无法直接同步（需要 kernel 边界或原子操作）

### 4. SIMT 执行模型

**SIMT = Single Instruction, Multiple Threads**

- 一个 Warp 中的 32 个线程在同一时钟周期执行**相同的指令**
- 但每个线程有**独立的寄存器**和**执行状态**
- 类似 CPU 的 SIMD，但更灵活（每个线程有独立 PC）

**Warp Divergence（分支分叉）**：

```c
// 代码中有 if-else
if (threadIdx.x % 2 == 0) {
    // 偶数线程执行路径A
    do_something_A();
} else {
    // 奇数线程执行路径B
    do_something_B();
}
```

执行过程：
```
Warp 执行时序：
时钟周期 1-10：所有32线程执行路径A（奇数线程被 mask 掉）
时钟周期 11-20：所有32线程执行路径B（偶数线程被 mask 掉）
总耗时：20 个周期 vs 理想的 10 个周期（效率损失50%）
```

**避免 Warp Divergence 的原则**：
- 让同一 Warp 内的线程（threadIdx.x % 32 相同的线程）走相同的分支
- 数据排布尽量让同 Warp 线程访问相似数据

### 5. Occupancy（占用率）

**Occupancy = 实际驻留 Warp 数 / SM 最大支持 Warp 数**

SM 资源限制（A100 示例）：
- 每个 SM 最大 Warp 数：64
- 每个 SM 最大 Block 数：32
- 每个 SM 寄存器总量：65536 个（256KB / 4字节）
- 每个 SM 共享内存：192KB（可配置）

**Occupancy 计算示例**：

假设 kernel：blockDim=256（=8 Warp），每个线程使用 32 个寄存器

```
寄存器限制：65536 / (256线程 × 32寄存器) = 8 个 Block
           = 8 × 8 Warp = 64 Warp（刚好100% Occupancy）

如果每线程用 64 个寄存器：
65536 / (256 × 64) = 4 个 Block = 32 Warp（50% Occupancy）
```

**高 Occupancy ≠ 高性能**：Occupancy 是隐藏内存延迟的手段，但如果 kernel 是 compute-bound，100% Occupancy 不会带来额外提升。

### 6. 内存访问模式：Coalesced Access

**Coalesced（合并）内存访问**：Warp 内 32 个线程同时访问连续的内存地址，GPU 硬件将其合并为少数几次内存事务。

```
线程 0: 访问 addr + 0
线程 1: 访问 addr + 4
线程 2: 访问 addr + 8
...
线程 31: 访问 addr + 124

→ 合并为 1 次 128 字节的内存事务（FP32情况）✅
```

**非合并（Strided/Random）访问**：

```
线程 0: 访问 addr + 0
线程 1: 访问 addr + 128   ← 不连续！
线程 2: 访问 addr + 256
...
→ 需要 32 次独立内存事务，带宽效率 1/32 ❌
```

**实践规则**：让 `threadIdx.x` 对应最内层（连续）的数组维度。

---

## 🖼️ 原理图解

### SM 内部结构（简化）

```
┌──────────────────────────────────────────────────────────┐
│                    Streaming Multiprocessor               │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐                      │
│  │ Warp Sched 0 │  │ Warp Sched 1 │  ← 每周期选1个Warp   │
│  └──────┬───────┘  └──────┬───────┘                      │
│         ↓                 ↓                               │
│  ┌──────────────────────────────────────────────────┐    │
│  │  CUDA Cores (FP32 × 32)  | Tensor Cores × 1      │    │
│  ├──────────────────────────────────────────────────┤    │
│  │  CUDA Cores (FP32 × 32)  | Tensor Cores × 1      │    │
│  └──────────────────────────────────────────────────┘    │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │           寄存器文件 (256KB)                        │  │
│  │     [T0][T1][T2]...[T255]  每个线程独有寄存器       │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │    L1 Cache + 共享内存 (可配置，最大164KB)           │  │
│  │    [Block0 Shared Mem] | [Block1 Shared Mem] | ...  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Grid / Block / Warp / Thread 层级

```
Grid (gridDim = 4×2)
┌────────────────────────────┐
│ Block(0,0)  Block(1,0)  Block(2,0)  Block(3,0) │
│ Block(0,1)  Block(1,1)  Block(2,1)  Block(3,1) │
└────────────────────────────┘

Block(0,0) (blockDim = 128)
├── Warp 0:  Thread  0 ~ 31
├── Warp 1:  Thread 32 ~ 63
├── Warp 2:  Thread 64 ~ 95
└── Warp 3:  Thread 96 ~ 127
```

---

## 🛠️ 动手练习

### 练习 1：查看 SM 规格

```python
import torch

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}")
    print(f"SM 数量: {props.multi_processor_count}")
    print(f"每 SM 最大线程数: {props.max_threads_per_multi_processor}")
    print(f"每 Block 最大线程数: {props.max_threads_per_block}")
    print(f"每 SM 共享内存: {props.shared_memory_per_multiprocessor / 1024:.0f} KB")
    print(f"每 Block 共享内存: {props.shared_memory_per_block / 1024:.0f} KB")
    print(f"寄存器文件大小: {props.regs_per_multiprocessor / 1024:.0f} K 寄存器")
    
    # 计算理论 Occupancy
    regs_per_thread = 32  # 假设值
    threads_per_block = 256
    max_blocks = min(
        props.max_blocks_per_multi_processor,
        props.regs_per_multiprocessor // (threads_per_block * regs_per_thread),
        props.shared_memory_per_multiprocessor // 1  # 假设无共享内存占用
    )
    warps_in_use = max_blocks * (threads_per_block // 32)
    max_warps = props.max_threads_per_multi_processor // 32
    print(f"\n理论 Occupancy: {warps_in_use}/{max_warps} = {warps_in_use/max_warps:.1%}")
```

### 练习 2：Warp Divergence 实验

```c
// 文件：warp_divergence.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel_no_divergence(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 所有线程同一路径
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

__global__ void kernel_with_divergence(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Warp 内线程走不同路径
        if (threadIdx.x % 2 == 0) {
            for (int i = 0; i < 100; i++) data[idx] = data[idx] * 2.0f;
        } else {
            for (int i = 0; i < 100; i++) data[idx] = data[idx] + 1.0f;
        }
    }
}

int main() {
    int n = 1 << 20;
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    
    dim3 block(256), grid((n + 255) / 256);
    
    // 对比两个 kernel 的耗时
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) kernel_no_divergence<<<grid, block>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_no_div;
    cudaEventElapsedTime(&ms_no_div, start, stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) kernel_with_divergence<<<grid, block>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_div;
    cudaEventElapsedTime(&ms_div, start, stop);
    
    printf("无分叉: %.2f ms\n有分叉: %.2f ms\n慢了: %.2fx\n",
           ms_no_div, ms_div, ms_div/ms_no_div);
    
    cudaFree(d_data);
    return 0;
}
```

```bash
nvcc -O2 -o warp_divergence warp_divergence.cu
./warp_divergence
```

### 练习 3：理解 Occupancy 工具

```bash
# 使用 Nsight Compute 分析（如果已安装）
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./your_program

# 或使用 nvprof（旧版）
nvprof --metrics achieved_occupancy ./your_program
```

---

## 📝 小结

| 概念 | 关键数字/规则 |
|------|-------------|
| Warp 大小 | 固定 32 个线程（NVIDIA GPU 历史传承） |
| SM 调度单位 | Warp（不是线程，不是 Block） |
| Warp Divergence | 同 Warp 内不同路径 → 串行执行，效率降低 |
| 共享内存 | Block 内线程共享，比全局内存快 ~100x |
| Occupancy 影响因素 | 寄存器用量、共享内存用量、Block 大小 |
| 合并内存访问 | 连续访问 → 1次事务；随机访问 → 32次事务 |

**关键认知**：SM 是理解 CUDA 性能的核心。你写的每一行 CUDA 代码，最终都在 SM 内的 Warp 调度器视角下执行。理解 SM 资源约束，是优化 CUDA kernel 的基础。

---

**明日预告**：D3 将学习 CUDA 编程模型——掌握 Thread/Block/Grid 的编程方式，理解内存空间（全局内存/共享内存/寄存器）的使用规则。
