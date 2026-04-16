# D3：CUDA 编程模型——从零认识并行计算框架

> Week1 主题：GPU架构与CUDA基础 | AI Infra线

---

## 🎯 学习目标

- 理解 CUDA 编程的核心抽象：Host/Device、Kernel、内存空间
- 掌握 CUDA 的内存管理 API：`cudaMalloc`、`cudaMemcpy`、`cudaFree`
- 理解 Grid/Block/Thread 的索引计算方式
- 能独立编写并编译一个简单的 CUDA 程序

---

## 🧠 核心知识点

### 1. CUDA 编程的基本概念

**Host vs Device**：
- **Host**：CPU 及其内存（主存 DRAM）
- **Device**：GPU 及其内存（显存 HBM/GDDR）

**Kernel（核函数）**：在 Device 上并行执行的函数
- 用 `__global__` 修饰符声明
- 由 Host 调用，在 Device 上执行
- 语法：`kernel_name<<<gridDim, blockDim>>>(args...)`

```c
// Host 代码（普通 C/C++）
void host_function() {
    // 在 CPU 上执行
    launch_kernel<<<grid, block>>>(args);  // 启动 GPU kernel
}

// Device 代码（CUDA C/C++）
__global__ void my_kernel(float* data, int n) {
    // 在 GPU 上并行执行（每个线程执行一次）
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}
```

### 2. 函数限定符（Function Qualifiers）

| 限定符 | 执行位置 | 调用位置 | 说明 |
|--------|---------|---------|------|
| `__global__` | Device | Host | Kernel 函数，返回值必须是 void |
| `__device__` | Device | Device | 只能从 Device 调用的辅助函数 |
| `__host__` | Host | Host | 普通 CPU 函数（默认，可省略） |
| `__host__ __device__` | 两者 | 两者 | 同时编译两个版本（如数学辅助函数） |

### 3. 线程索引计算

CUDA 提供以下内建变量（在 kernel 内使用）：

```c
// 描述 Block 在 Grid 中的位置（最多3维）
blockIdx.x, blockIdx.y, blockIdx.z

// 描述 Thread 在 Block 中的位置（最多3维）
threadIdx.x, threadIdx.y, threadIdx.z

// Block 的尺寸（一个 Block 中有多少线程）
blockDim.x, blockDim.y, blockDim.z

// Grid 的尺寸（一个 Grid 中有多少 Block）
gridDim.x, gridDim.y, gridDim.z
```

**1D 索引计算（最常见）**：

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// 理解：Block 编号 × Block大小 + Block内线程偏移
```

**2D 索引计算（处理矩阵时常用）**：

```c
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * width + col;  // 行主序（C 风格）
```

**启动配置计算**：

```c
int n = 1000000;
int block_size = 256;                             // 每 Block 线程数（推荐256或128）
int grid_size = (n + block_size - 1) / block_size; // 向上取整

kernel<<<grid_size, block_size>>>(data, n);
```

### 4. CUDA 内存空间

| 内存类型 | 声明方式 | 作用域 | 生命周期 | 速度 |
|---------|---------|-------|---------|------|
| 寄存器 | 自动（局部变量） | 线程私有 | Kernel | 最快 |
| 局部内存 | 自动（数组/溢出） | 线程私有 | Kernel | 慢（在全局内存） |
| 共享内存 | `__shared__` | Block 内共享 | Block | 快（~L1） |
| 全局内存 | `cudaMalloc` | 所有线程 | Application | 慢（HBM） |
| 常量内存 | `__constant__` | 只读，所有线程 | Application | 有缓存，小数据快 |
| 纹理内存 | `tex1D` 等 | 只读，所有线程 | Application | 有缓存，空间局部性好 |

**共享内存示例**：

```c
__global__ void shared_mem_example(float* in, float* out, int n) {
    __shared__ float tile[256];  // 静态分配共享内存
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;
    
    // 从全局内存加载到共享内存
    if (idx < n) tile[local_idx] = in[idx];
    __syncthreads();  // 等待所有线程加载完成
    
    // 在共享内存上操作（快！）
    if (idx < n && local_idx > 0) {
        out[idx] = tile[local_idx] + tile[local_idx - 1];
    }
}
```

### 5. CUDA 内存管理 API

```c
// 1. 在 Device 上分配内存
float* d_data;
cudaMalloc((void**)&d_data, n * sizeof(float));

// 2. 将数据从 Host 拷贝到 Device
cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

// 3. 将数据从 Device 拷贝到 Host
cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

// 4. 释放 Device 内存
cudaFree(d_data);

// 5. 错误检查（重要！）
cudaError_t err = cudaMalloc(...);
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
}
```

**常用宏（推荐加入项目）**：

```c
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// 使用示例
CUDA_CHECK(cudaMalloc((void**)&d_data, n * sizeof(float)));
CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice));
```

### 6. CUDA 同步机制

```c
// Kernel 调用是异步的！
kernel<<<grid, block>>>(args);
// 此时 CPU 不等待 GPU 完成，继续执行

// 等待 GPU 完成所有操作
cudaDeviceSynchronize();

// Block 内同步
__syncthreads();  // 只能在 kernel 内使用，等待 Block 内所有线程到达此点

// 使用 CUDA Event 精确计时
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(args);
cudaEventRecord(stop);
cudaEventSynchronize(stop);  // 等待 stop event 完成

float milliseconds;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Kernel 耗时: %.3f ms\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

### 7. CUDA 程序的执行流程

```
Host (CPU)                          Device (GPU)
  │                                     │
  ├─ 分配 Host 内存 (malloc)             │
  ├─ 初始化数据                          │
  ├─ 分配 Device 内存 (cudaMalloc) ──→  │ (GPU显存分配)
  ├─ 拷贝数据 H→D (cudaMemcpy) ──────→  │ (数据传输, PCIe)
  ├─ 启动 Kernel ──────────────────────→│ (异步执行)
  │   [CPU 可继续做其他事情]             ├─ Grid 调度到各 SM
  │                                     ├─ Block 分配到 SM
  │                                     ├─ Warp 执行计算
  │                                     │  ...
  ├─ cudaDeviceSynchronize() ─ 等待 ───→│ (等待完成)
  ├─ 拷贝数据 D→H (cudaMemcpy) ←──────  │ (结果传回)
  ├─ 处理结果                            │
  └─ 释放内存                            │
```

---

## 🖼️ 原理图解

### Grid-Block-Thread 三级层次

```
Kernel Launch: <<<gridDim=(4,2), blockDim=(4,4)>>>

Grid (4×2 个 Block):
┌──────────────────────────────────────────┐
│ B(0,0) B(1,0) B(2,0) B(3,0)             │
│ B(0,1) B(1,1) B(2,1) B(3,1)             │
└──────────────────────────────────────────┘

Block B(1,0) (4×4=16个线程):
┌─────────────────────────┐
│ T(0,0) T(1,0) T(2,0) T(3,0) │  ← Warp 0: 线程 0~3 (如果是1D则16线程/warp不成立)
│ T(0,1) T(1,1) T(2,1) T(3,1) │
│ T(0,2) T(1,2) T(2,2) T(3,2) │
│ T(0,3) T(1,3) T(2,3) T(3,3) │
└─────────────────────────┘

线程 T(2,1) 的全局索引（按行主序展平）:
  row = blockIdx.y * blockDim.y + threadIdx.y = 0*4 + 1 = 1
  col = blockIdx.x * blockDim.x + threadIdx.x = 1*4 + 2 = 6
```

### 内存访问模式示意

```
全局内存（慢，TB/s 量级）
    ↕ 约 200-400 个时钟周期延迟
L2 Cache（SM 间共享）
    ↕ 约 30-100 个时钟周期延迟
L1 Cache + 共享内存（SM 内）
    ↕ 约 4-30 个时钟周期延迟
寄存器（每个线程私有）
    ↕ 1 个时钟周期
```

---

## 🛠️ 动手练习

### 练习 1：CUDA 环境验证

```bash
# 查看 CUDA 版本
nvcc --version
nvidia-smi

# 查看设备信息
cat << 'EOF' > device_info.cu
#include <cuda_runtime.h>
#include <stdio.h>
int main() {
    int count;
    cudaGetDeviceCount(&count);
    printf("发现 %d 个 GPU\n", count);
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("GPU %d: %s\n", i, prop.name);
        printf("  SM 数量: %d\n", prop.multiProcessorCount);
        printf("  每SM最大线程: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  每Block最大线程: %d\n", prop.maxThreadsPerBlock);
        printf("  全局内存: %.0f MB\n", prop.totalGlobalMem / 1e6);
        printf("  每Block共享内存: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Warp大小: %d\n", prop.warpSize);
        printf("  CUDA 版本: %d.%d\n", prop.major, prop.minor);
    }
    return 0;
}
EOF
nvcc -o device_info device_info.cu && ./device_info
```

### 练习 2：向量加法（经典 Hello World）

```c
// vector_add.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1 << 20;  // 1M 元素
    size_t bytes = n * sizeof(float);
    
    // 分配 Host 内存
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(n - i);
    }
    
    // 分配 Device 内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // 拷贝数据 H→D
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // 计时并启动 Kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    cudaEventRecord(start);
    vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel 耗时: %.3f ms\n", ms);
    printf("有效带宽: %.1f GB/s\n", 3 * bytes / ms / 1e6);
    
    // 拷贝结果 D→H
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // 验证结果
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) errors++;
    }
    printf("验证结果: %d 个错误（应为0）\n", errors);
    
    // 清理
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
```

```bash
nvcc -O2 -o vector_add vector_add.cu && ./vector_add
```

### 练习 3：思考题

1. 如果 `n = 1000`，`block_size = 256`，Grid 中最后一个 Block 里有多少"有效"线程？多余的线程执行了什么？
2. 为什么 `vector_add` kernel 中需要 `if (idx < n)` 这个边界检查？
3. `cudaDeviceSynchronize()` 和 `cudaEventSynchronize()` 的区别是什么？

---

## 📝 小结

| 概念 | 说明 |
|------|------|
| Host/Device | CPU+主存 / GPU+显存，相互独立的地址空间 |
| Kernel | `__global__` 函数，在 GPU 并行执行 |
| 启动语法 | `kernel<<<grid, block>>>(args)` |
| 线程索引 | `blockIdx.x * blockDim.x + threadIdx.x` |
| 内存分配 | `cudaMalloc` / `cudaFree`（Device 侧） |
| 数据传输 | `cudaMemcpy`（PCIe 瓶颈，尽量减少） |
| 同步 | Kernel 异步启动，`cudaDeviceSynchronize()` 等待 |
| 内存层级 | 寄存器 > 共享内存 > L2 > 全局内存 |

**关键认知**：CUDA 编程的核心心智模型是"大量独立线程并行执行同一段代码（Kernel），每个线程通过内建变量计算自己的工作范围"。理解线程索引和内存管理，是写出正确 CUDA 代码的前提。

---

**明日预告**：D4 将动手写第一个真正的 CUDA Kernel——矩阵转置，深入理解合并内存访问与共享内存的配合使用。
