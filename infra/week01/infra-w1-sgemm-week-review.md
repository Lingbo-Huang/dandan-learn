# D7：Week1 综合实战——实现 SGEMM 并与 cuBLAS 对比

> Week1 主题：GPU架构与CUDA基础 | AI Infra线

---

## 🎯 学习目标

- 综合运用本周知识，从零实现 SGEMM（单精度通用矩阵乘法）
- 理解矩阵乘法 Tiling 策略：从 Naive → Shared Memory Tiling → Register Tiling
- 与 cuBLAS 基准对比，理解"工业级"优化与手写 Kernel 的差距
- 完成 Week1 知识体系的自测与回顾

---

## 🧠 核心知识点

### 1. SGEMM 问题定义

计算 `C = A × B`，其中：
- A: [M, K]
- B: [K, N]  
- C: [M, N]

计算量：`2 × M × K × N` FLOPs（每个输出元素需要 K 次乘法 + K 次加法）

```
朴素算法：
for i in range(M):
    for j in range(N):
        for k in range(K):
            C[i][j] += A[i][k] * B[k][j]
```

### 2. Version 1：朴素 CUDA SGEMM

```c
__global__ void sgemm_naive(float* A, float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 启动配置
dim3 block(32, 32);
dim3 grid((N + 31) / 32, (M + 31) / 32);
sgemm_naive<<<grid, block>>>(A, B, C, M, K, N);
```

**问题分析**：
- 每个线程独立计算 K 次内积
- 读 A：线程 (row, col) 读整行 A[row, 0:K] → 不同线程读不同行 → 每次读全局内存
- 读 B：线程 (row, col) 读整列 B[0:K, col] → 不合并（列读取）
- 算术强度 ≈ 2K / (4 × 2K / 32) ≈ 16 （取决于缓存命中）

### 3. Version 2：共享内存 Tiling（SMEM SGEMM）

**核心思路**：将 A 和 B 的小块（Tile）加载到共享内存，在共享内存上做计算，减少全局内存访问。

```c
#define TILE_SIZE 32

__global__ void sgemm_tiled(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // 沿 K 维度分 Tile 计算
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; t++) {
        // Phase 1：合并加载 A 和 B 的 Tile 到共享内存
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        
        __syncthreads();  // 等待所有线程加载完成
        
        // Phase 2：在共享内存上计算（快！）
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();  // 等待所有线程计算完成，再加载下一个 Tile
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**性能分析**：
- 每个 Tile 从全局内存读 2 × TILE_SIZE² 个元素
- 每个 Tile 在共享内存上做 TILE_SIZE³ 次 FMA
- 算术强度提升：TILE_SIZE/2 ≈ 16 FLOPs/Byte（vs 朴素版本）

### 4. Version 3：寄存器 Tiling（更高算术强度）

让每个线程负责 `BM × BN` 个输出元素，把累加结果放在寄存器中：

```c
#define BM 64    // Block 覆盖的输出行数
#define BN 64    // Block 覆盖的输出列数
#define BK 16    // K 维度的 Tile 大小
#define TM 8     // 每个线程负责的输出行数
#define TN 8     // 每个线程负责的输出列数

__global__ void sgemm_reg_tiled(float* A, float* B, float* C, int M, int K, int N) {
    // Block 内线程数：(BM/TM) × (BN/TN) = 8 × 8 = 64
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];
    
    float regC[TM][TN] = {0.0f};  // 寄存器累加（TM×TN个输出）
    float regA[TM], regB[TN];     // 寄存器缓存
    
    int thread_row = threadIdx.x / (BN / TN);  // Block 内线程行
    int thread_col = threadIdx.x % (BN / TN);  // Block 内线程列
    
    int num_tiles = (K + BK - 1) / BK;
    for (int t = 0; t < num_tiles; t++) {
        // 加载 As, Bs 到共享内存（合并访问）
        // ... (省略加载代码)
        __syncthreads();
        
        // 从共享内存到寄存器，计算 TM×TN 的小矩阵乘
        for (int k = 0; k < BK; k++) {
            for (int m = 0; m < TM; m++) regA[m] = As[k][thread_row * TM + m];
            for (int n = 0; n < TN; n++) regB[n] = Bs[k][thread_col * TN + n];
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    regC[m][n] += regA[m] * regB[n];
        }
        __syncthreads();
    }
    
    // 写回全局内存
    // ...
}
```

**为什么更快**：
- 每个线程读 TM+TN 个元素 → 做 TM×TN 次 FMA
- 算术强度 = 2×TM×TN / ((TM+TN)×4) ≈ 2×8×8/(16×4) = 2 FLOPs/Byte（共享内存层）
- 实际上最内层完全在寄存器中，算术强度极高

### 5. cuBLAS 对比

```c
#include <cublas_v2.h>

void cublas_sgemm(float* A, float* B, float* C, int M, int K, int N) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f, beta = 0.0f;
    
    // cuBLAS 使用列主序（Fortran风格）
    // C_col = B_col^T × A_col^T
    // 用行主序传参时，等价于：C = A × B
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,   // 不转置
        N, M, K,                      // cuBLAS 的 n, m, k（注意顺序！）
        &alpha,
        B, N,                         // B 矩阵（cuBLAS 角度的 A）
        A, K,                         // A 矩阵（cuBLAS 角度的 B）
        &beta,
        C, N                          // C 矩阵
    );
    
    cublasDestroy(handle);
}
```

---

## 🖼️ 原理图解

### Tiling 策略图示

```
矩阵乘法 C[M][N] = A[M][K] × B[K][N]

朴素（每线程处理1个输出）：
  Thread(i,j) 计算 C[i][j]:
    读 A 的第 i 行（K 个元素）
    读 B 的第 j 列（K 个元素）
  共享内存复用：无
  全局内存访问：M×N×2K×4 字节

共享内存 Tiling（Block处理32×32块）：
  Block 负责 C[brow*32:(brow+1)*32][bcol*32:(bcol+1)*32]
  分 K/32 轮：每轮加载 32×32 的 A-tile 和 B-tile 到共享内存
  
  A-tile: [32][32] = 4KB 共享内存
  B-tile: [32][32] = 4KB 共享内存
  
  全局内存访问：M×N/32×32×2K×4/32 = M×N×2K×4/32 字节
  → 减少 32× 全局内存读取！

寄存器 Tiling（每线程处理8×8输出）：
  每个线程维护 8×8 个寄存器累加器
  每轮从共享内存读 8+8=16 个元素，做 64 次 FMA
  → 算术强度进一步提升
```

### 性能瀑布图（典型结果）

```
性能 (TFLOPS) 对比（A100, M=N=K=4096）

cuBLAS:         ████████████████████ 18.0 TFLOPS (~90% 峰值)
Register Tile:  ████████████████     14.0 TFLOPS (~70% 峰值)
SMEM Tile:      ████████             7.0  TFLOPS (~35% 峰值)
Naive:          ██                   2.0  TFLOPS (~10% 峰值)

峰值 FP32 A100: 19.5 TFLOPS
```

---

## 🛠️ 动手练习

### 完整 SGEMM 对比实验

```c
// sgemm_benchmark.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// --- 在此粘贴 sgemm_naive 和 sgemm_tiled 的代码 ---

void benchmark_kernel(const char* name, void (*kernel_launcher)(),
                      float* d_C, float* h_ref, int M, int N, int repeats) {
    // Warmup
    for (int i = 0; i < 3; i++) kernel_launcher();
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) kernel_launcher();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= repeats;
    
    float flops = 2.0f * M * 4096 * N;  // K = 4096
    float tflops = flops / ms / 1e9;
    
    // 验证正确性
    float* h_result = (float*)malloc(M * N * sizeof(float));
    cudaMemcpy(h_result, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    float max_err = 0;
    for (int i = 0; i < M * N; i++) {
        max_err = fmax(max_err, fabs(h_result[i] - h_ref[i]));
    }
    free(h_result);
    
    printf("%-20s | %8.3f ms | %6.2f TFLOPS | 最大误差: %.2e %s\n",
           name, ms, tflops, max_err, max_err < 1e-2 ? "✅" : "❌");
    
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    int M = 4096, K = 4096, N = 4096;
    
    // 初始化数据
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 用 cuBLAS 生成参考结果
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    float* h_ref = (float*)malloc(M * N * sizeof(float));
    cudaMemcpy(h_ref, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("%-20s | %-10s | %-14s | 正确性\n", "Kernel", "耗时", "算力");
    printf("%-20s-|-%-10s-|-%-14s-|--------\n", "--------------------", "----------", "--------------");
    
    int repeats = 20;
    
    // 测试各版本
    benchmark_kernel("cuBLAS", [&]() {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }, d_C, h_ref, M, N, repeats);
    
    dim3 block_naive(32, 32), grid_naive((N+31)/32, (M+31)/32);
    benchmark_kernel("Naive SGEMM", [&]() {
        sgemm_naive<<<grid_naive, block_naive>>>(d_A, d_B, d_C, M, K, N);
    }, d_C, h_ref, M, N, repeats);
    
    dim3 block_tile(32, 32), grid_tile((N+31)/32, (M+31)/32);
    benchmark_kernel("SMEM Tiled SGEMM", [&]() {
        sgemm_tiled<<<grid_tile, block_tile>>>(d_A, d_B, d_C, M, K, N);
    }, d_C, h_ref, M, N, repeats);
    
    free(h_A); free(h_B); free(h_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);
    return 0;
}
```

```bash
nvcc -O2 -o sgemm_bench sgemm_benchmark.cu -lcublas && ./sgemm_bench
```

### Week1 自测题

**概念回顾**（每题2分，共20分）：

1. GPU 的 SM 数量与 CUDA Core 数量是什么关系？（以 H100 为例）
2. Warp 的大小是多少？为什么是这个数字（历史原因）？
3. 什么是 Warp Divergence？如何避免？
4. `cudaMemcpy` 的四种方向分别是什么？
5. `__syncthreads()` 的作用范围是什么（Block? Grid? Warp?）？
6. 合并内存访问的条件是什么？
7. 什么是 Bank Conflict？如何用 padding 解决？
8. Occupancy 的定义是什么？影响 Occupancy 的三个主要因素？
9. Nsight Systems 和 Nsight Compute 的主要区别？
10. SGEMM 中，共享内存 Tiling 的目的是什么？

**编程挑战**（选做）：
- [ ] 实现一个 CUDA Kernel：对长度 n 的数组求最大值（用 Warp Shuffle）
- [ ] 实现 2D 卷积 Kernel（stride=1, padding=0），与 PyTorch 对比结果
- [ ] 用 ncu 分析你写的 SGEMM，找出 Memory Throughput 和 Compute Throughput

---

## 📝 Week1 总结与知识导图

### 本周知识体系

```
Week1: GPU架构与CUDA基础
│
├── D1: GPU vs CPU
│   ├── 设计哲学：延迟导向 vs 吞吐导向
│   ├── 核心数量、内存带宽的数量级差异
│   └── Roofline 模型：判断算力/内存瓶颈
│
├── D2: SM 架构
│   ├── Warp / Thread / Block / Grid 层级
│   ├── SIMT 执行模型与 Warp Divergence
│   ├── Occupancy：资源约束与占用率
│   └── 合并内存访问（Coalesced Access）
│
├── D3: CUDA 编程模型
│   ├── Host / Device / Kernel 概念
│   ├── 内存空间：寄存器/共享内存/全局内存
│   ├── CUDA API：cudaMalloc/cudaMemcpy/cudaFree
│   └── 线程索引计算
│
├── D4: 第一个 CUDA Kernel（矩阵转置）
│   ├── 合并访问 vs 非合并访问的性能差距
│   ├── 共享内存 Tiling 解决非合并问题
│   └── Bank Conflict 与 Padding 技巧
│
├── D5: 内存层次与归约
│   ├── 并行归约的 6 个版本
│   ├── Warp Shuffle：__shfl_down_sync
│   └── 归约是 LayerNorm/Softmax 的核心
│
├── D6: 性能分析工具
│   ├── nsys: 系统时间线、传输瓶颈
│   ├── ncu: 单 Kernel 硬件指标、Roofline
│   └── torch.profiler: PyTorch 算子分析
│
└── D7: 综合实战（SGEMM）
    ├── Naive → SMEM Tiling → 寄存器 Tiling
    ├── 与 cuBLAS 性能对比
    └── Week1 自测
```

### 下周预告：Week2——深度学习框架原理

- D1: PyTorch autograd 机制（计算图、梯度计算）
- D2: 算子（Operator）注册与分发机制
- D3: Dispatcher、DispatchKey、设备抽象
- D4: TorchScript 与 FX 图
- D5: 内存管理（CachingAllocator）
- D6: DataLoader 与数据流水线
- D7: 综合实战：从零写一个自定义算子

---

## 📝 小结

本周从 GPU 与 CPU 的设计哲学出发，深入到 SM 架构、CUDA 编程模型，并通过矩阵转置、归约、SGEMM 三个经典案例，建立了完整的 CUDA 性能优化认知体系：

| 核心原则 | 说明 |
|---------|------|
| 数据并行 | GPU 的核心优势，让尽量多的线程同时工作 |
| 合并访问 | 内存带宽利用率的关键，相邻线程访问相邻地址 |
| 共享内存 | 比全局内存快100x，是优化内存密集 Kernel 的首选 |
| 避免分叉 | 同 Warp 走相同分支，防止 SIMT 串行化 |
| 先 Profile | 用 nsys/ncu 定位真实瓶颈，再针对性优化 |

**关键认知**：理解 CUDA 并不是终点，而是理解 cuDNN、PyTorch、TensorRT 这些上层系统的基础。Week2 我们将进入 PyTorch 内部，看这些框架是如何把 CUDA Kernel 组织成一个高效的深度学习计算引擎的。
