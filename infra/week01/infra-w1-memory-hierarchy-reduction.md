# D5：CUDA 内存层次与归约优化——从朴素到高效

> Week1 主题：GPU架构与CUDA基础 | AI Infra线

---

## 🎯 学习目标

- 深入理解 CUDA 内存层次：寄存器/共享内存/L1/L2/全局内存的特性
- 掌握 Parallel Reduction（并行归约）的标准优化路径
- 理解 Warp-level 原语（`__shfl_down_sync`）
- 能用 6 个步骤将朴素归约优化到接近带宽上限

---

## 🧠 核心知识点

### 1. 内存层次完整回顾

| 内存类型 | 大小（A100/H100） | 延迟 | 带宽 | 作用域 |
|---------|-----------------|------|------|--------|
| 寄存器 | 256KB/SM，每线程最多255个 | 1 cycle | N/A | 线程私有 |
| L1/共享内存 | 192KB/SM（可配） | 28 cycles | ~19 TB/s | Block/SM |
| L2 Cache | 40MB~50MB | 200 cycles | ~12 TB/s | 全 GPU |
| HBM 全局内存 | 80GB | 600+ cycles | 3.35 TB/s | 全 GPU |

**共享内存 vs L1 的配置（Ampere）**：
```
总量：192KB per SM
可配置比例：
  L1:Smem = 128KB:64KB
  L1:Smem = 96KB:96KB
  L1:Smem = 28KB:164KB  ← 共享内存优先模式
  
cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 75);
// 75% 分配给共享内存
```

### 2. 并行归约：问题定义

**求和归约（Sum Reduction）**：
```
输入：[a0, a1, a2, ..., a_{n-1}]
输出：a0 + a1 + a2 + ... + a_{n-1}

朴素 CPU：O(n) 顺序加法
GPU 目标：O(log n) 并行归约，O(n/p) 工作 per 线程
```

### 3. 归约优化的 6 个版本

#### Version 1: 朴素归约（Interleaved Addressing，有分叉）

```c
__global__ void reduce_v1(int* g_in, int* g_out, int n) {
    extern __shared__ int s_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    s_data[tid] = (idx < n) ? g_in[idx] : 0;
    __syncthreads();
    
    // 交错寻址：步长为2的幂次递增
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {  // ❌ Warp Divergence
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = s_data[0];
}
```

**问题**：`tid % (2*stride) == 0` 导致 Warp Divergence，且地址计算低效

#### Version 2: 消除 Warp Divergence（Sequential Addressing）

```c
__global__ void reduce_v2(int* g_in, int* g_out, int n) {
    extern __shared__ int s_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    s_data[tid] = (idx < n) ? g_in[idx] : 0;
    __syncthreads();
    
    // 步长从大到小，相邻归约 → 无分叉
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {  // ✅ 前一半线程工作，整个 Warp 走同一路径
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = s_data[0];
}
```

**改进**：消除分叉，性能提升约 2x

#### Version 3: 减少空闲线程（First Add During Load）

```c
__global__ void reduce_v3(int* g_in, int* g_out, int n) {
    extern __shared__ int s_data[];
    int tid = threadIdx.x;
    // 每个 Block 处理 2*blockDim.x 的数据（减少 Block 数量，减少线程空转）
    int idx = blockIdx.x * (blockDim.x * 2) + tid;
    
    // 加载时同时做第一次加法
    s_data[tid] = (idx < n ? g_in[idx] : 0) + 
                  (idx + blockDim.x < n ? g_in[idx + blockDim.x] : 0);
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_data[tid] += s_data[tid + stride];
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = s_data[0];
}
```

**改进**：减少 Block 数量，充分利用每个线程的工作量

#### Version 4: Warp Unrolling（最后一个 Warp 无需同步）

```c
__device__ void warp_reduce(volatile int* s_data, int tid) {
    // 最后 32 个线程（一个 Warp）：不需要 __syncthreads__（同一 Warp 自然同步）
    s_data[tid] += s_data[tid + 32];
    s_data[tid] += s_data[tid + 16];
    s_data[tid] += s_data[tid + 8];
    s_data[tid] += s_data[tid + 4];
    s_data[tid] += s_data[tid + 2];
    s_data[tid] += s_data[tid + 1];
}

__global__ void reduce_v4(int* g_in, int* g_out, int n) {
    extern __shared__ int s_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + tid;
    s_data[tid] = (idx < n ? g_in[idx] : 0) + 
                  (idx + blockDim.x < n ? g_in[idx + blockDim.x] : 0);
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) s_data[tid] += s_data[tid + stride];
        __syncthreads();
    }
    if (tid < 32) warp_reduce(s_data, tid);  // ✅ 最后32个线程特殊处理
    if (tid == 0) g_out[blockIdx.x] = s_data[0];
}
```

#### Version 5: 完全展开（模板 + 编译时展开）

```c
template <unsigned int blockSize>
__global__ void reduce_v5(int* g_in, int* g_out, int n) {
    extern __shared__ int s_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockSize * 2) + tid;
    s_data[tid] = (idx < n ? g_in[idx] : 0) + 
                  (idx + blockSize < n ? g_in[idx + blockSize] : 0);
    __syncthreads();
    
    // 编译时展开，消除循环开销
    if (blockSize >= 512) { if (tid < 256) s_data[tid] += s_data[tid+256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) s_data[tid] += s_data[tid+128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) s_data[tid] += s_data[tid+ 64]; __syncthreads(); }
    if (tid < 32) {
        if (blockSize >= 64) s_data[tid] += s_data[tid+32];
        if (blockSize >= 32) s_data[tid] += s_data[tid+16];
        if (blockSize >= 16) s_data[tid] += s_data[tid+ 8];
        if (blockSize >=  8) s_data[tid] += s_data[tid+ 4];
        if (blockSize >=  4) s_data[tid] += s_data[tid+ 2];
        if (blockSize >=  2) s_data[tid] += s_data[tid+ 1];
    }
    if (tid == 0) g_out[blockIdx.x] = s_data[0];
}
```

#### Version 6: Warp Shuffle（无需共享内存，最优）

```c
__inline__ __device__ int warp_reduce_sum(int val) {
    // __shfl_down_sync：Warp 内线程之间直接交换寄存器值
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_v6(int* g_in, int* g_out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;   // Warp 内线程 ID
    int warpId = threadIdx.x / 32; // Block 内 Warp ID
    
    // 每个线程加载并做初始归约
    int val = (tid < n) ? g_in[tid] : 0;
    
    // Warp 内归约（用 shuffle，不需要共享内存）
    val = warp_reduce_sum(val);
    
    // 各 Warp 的结果写入共享内存（只需要 blockDim.x/32 个槽位）
    __shared__ int warp_sums[32];
    if (lane == 0) warp_sums[warpId] = val;
    __syncthreads();
    
    // 第一个 Warp 对各 Warp 结果做最终归约
    val = (threadIdx.x < blockDim.x / 32) ? warp_sums[lane] : 0;
    if (warpId == 0) val = warp_reduce_sum(val);
    
    if (threadIdx.x == 0) g_out[blockIdx.x] = val;
}
```

**`__shfl_down_sync` 原理**：

```
Warp 中16个线程（简化）：
val: [8, 3, 5, 1, 2, 7, 4, 6, 9, 0, 3, 2, 1, 5, 8, 4]

offset=8: 每个线程 += 右边第8个线程的值
val: [17, 3, 8, 3, 11, 12, 12, 10, 9, 0, ...]

offset=4: 每个线程 += 右边第4个线程的值
...

最终：线程0的 val = 所有16个值的和
```

### 4. 性能对比总结

| 版本 | 关键优化 | 相对性能 |
|------|---------|---------|
| V1: 交错寻址 | 无 | 1x |
| V2: 顺序寻址 | 消除 Warp Divergence | 2x |
| V3: 加载时加法 | 减少空闲线程 | 3x |
| V4: Warp 展开 | 减少同步开销 | 4x |
| V5: 完全展开 | 消除循环指令 | 4.5x |
| V6: Warp Shuffle | 无共享内存，寄存器通信 | 5x+ |

---

## 🖼️ 原理图解

### 归约树（V2 顺序寻址）

```
初始: [1, 2, 3, 4, 5, 6, 7, 8]  (8个元素, 8个线程)

stride=4: 
  T0: s[0] += s[4]  → [6,  2, 3, 4,  -, -, -, -]
  T1: s[1] += s[5]  → [6,  8, 3, 4,  -, -, -, -]
  T2: s[2] += s[6]  → [6,  8,10, 4,  -, -, -, -]
  T3: s[3] += s[7]  → [6,  8,10,12,  -, -, -, -]

stride=2:
  T0: s[0] += s[2]  → [16, 8,10,12, ...]
  T1: s[1] += s[3]  → [16,20,10,12, ...]

stride=1:
  T0: s[0] += s[1]  → [36,20,10,12, ...]

结果: s[0] = 36 = 1+2+3+4+5+6+7+8 ✅
```

### Warp Shuffle 通信

```
__shfl_down_sync(mask, val, offset):
  线程 i 获得线程 (i + offset) 的 val 值
  
offset=4:
  T0 ← T4的值    T1 ← T5的值    T2 ← T6的值    T3 ← T7的值
  T4 ← T8的值    T5 ← T9的值    ...

→ 纯寄存器通信，无内存访问，无同步开销
```

---

## 🛠️ 动手练习

### 练习 1：实现并对比 V2 和 V6

```c
// reduction_benchmark.cu
// 实现 reduce_v2 和 reduce_v6，计算 100M 个整数的和，对比性能

#include <cuda_runtime.h>
#include <stdio.h>

// ... （将上述 kernel 代码粘贴到此处）

int main() {
    int n = 1 << 27;  // 128M 元素
    // 分配内存、初始化数据、对比两版本耗时
    // 验证结果与 CPU 计算的差异
}
```

```bash
nvcc -O2 -o reduction reduction_benchmark.cu && ./reduction
```

### 练习 2：理解 Warp Shuffle

```c
// 实验：用 __shfl_down_sync 计算一个 Warp 的最大值
__device__ int warp_reduce_max(int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other = __shfl_down_sync(0xffffffff, val, offset);
        val = max(val, other);
    }
    return val;
}

__global__ void find_warp_max(int* data, int* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int val = idx < n ? data[idx] : INT_MIN;
    val = warp_reduce_max(val);
    if (threadIdx.x % 32 == 0) result[blockIdx.x * 32 + threadIdx.x / 32] = val;
}
```

### 练习 3：用 PyTorch 验证

```python
import torch

# CPU 参考
data = torch.randint(0, 100, (1 << 20,), dtype=torch.int32)
cpu_sum = data.sum().item()

# GPU 计算
data_gpu = data.cuda()
gpu_sum = data_gpu.sum().item()

print(f"CPU sum: {cpu_sum}")
print(f"GPU sum: {gpu_sum}")
print(f"一致: {cpu_sum == gpu_sum}")
```

---

## 📝 小结

| 优化技术 | 原理 | 收益 |
|---------|------|------|
| 顺序寻址 | 消除 Warp Divergence | 约2x |
| 加载时归约 | 减少 Block 数/线程空转 | 约1.5x |
| Warp 展开 | 最后一个 Warp 无需同步 | 减少同步指令 |
| 编译时展开 | 减少运行时分支和循环 | 减少指令数 |
| Warp Shuffle | 寄存器直接通信，绕过共享内存 | 最快 |

**关键认知**：归约是深度学习中的基础算子（BatchNorm、Softmax、LayerNorm 内部都有归约）。理解归约优化，等于理解 GPU 上几乎所有高级算子的优化套路：**最大化内存带宽利用率、消除分叉、减少同步开销、使用 Warp-level 原语**。

---

**明日预告**：D6 将学习 CUDA 性能分析工具——Nsight Systems 和 Nsight Compute 的实际使用，学会读懂 kernel 性能瓶颈报告。
