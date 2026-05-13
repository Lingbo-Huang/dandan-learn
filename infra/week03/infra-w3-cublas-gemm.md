---
layout: default
title: "D1 · cuBLAS 架构与 GEMM 深度解析"
---

# D1 · cuBLAS 架构与 GEMM

> **AI Infra Week 3**  
> 神经网络 80% 以上的计算时间花在矩阵乘法上。cuBLAS 是 NVIDIA 为矩阵运算专门优化的库。

---

## 一、为什么需要 cuBLAS？

**矩阵乘法（GEMM）是深度学习的核心计算**：

```
全连接层：Y = X @ W  (GEMM)
Attention：Q @ K.T  (GEMM × 4)
Conv2D：展开后也是 GEMM（im2col）
```

自己写的 CUDA kernel 和 cuBLAS 的性能差距：

| 实现 | 性能（TFLOPS，A100） |
|------|-------------------|
| 朴素 CUDA（每线程一元素） | ~1 TFLOPS |
| 分块 Shared Memory | ~10 TFLOPS |
| Tensor Core 手动利用 | ~100 TFLOPS |
| **cuBLAS** | **~312 TFLOPS（接近峰值）** |

---

## 二、GEMM 的数学与计算挑战

$$C = \alpha \cdot A \times B + \beta \cdot C$$

- $A$：$M \times K$ 矩阵
- $B$：$K \times N$ 矩阵
- $C$：$M \times N$ 矩阵
- 计算量：$2MKN$ 次浮点运算

**朴素实现的问题**：

```c
// 朴素 CUDA GEMM（性能极差）
__global__ void naive_gemm(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];  // 每次都访问全局内存！
        }
        C[row * N + col] = sum;
    }
}
// 问题：每次读 A 和 B 都是全局内存访问（带宽受限）
```

---

## 三、Tiled GEMM：分块 + Shared Memory

```c
// 分块矩阵乘法（教学版）
#define TILE_SIZE 16

__global__ void tiled_gemm(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // 分 K/TILE_SIZE 轮，每轮加载一个 Tile 到 Shared Memory
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 协作加载 Tile（每个线程加载一个元素）
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();  // 等所有线程加载完
        
        // 用 Shared Memory 计算
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();  // 等所有线程计算完，再加载下一轮
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

**为什么快**：
- 每个 Tile 只从全局内存读取一次，之后从 Shared Memory（延迟约 4 倍速）访问
- `TILE_SIZE = 16`：每个元素被复用 16 次，全局内存访问减少 16 倍

---

## 四、Tensor Core：cuBLAS 性能的秘密

NVIDIA Volta 架构引入的 Tensor Core 专门做 4×4 矩阵乘法，每个时钟周期完成 64 次 FP16 乘加。

```
Tensor Core 一次操作：
[D 4×4] = [A 4×4] × [B 4×4] + [C 4×4]
64 次乘加 / 时钟周期 / Tensor Core

A100 有 432 个 Tensor Core
→ 432 × 64 = 27,648 次乘加/时钟
→ 1.4 GHz × 27,648 = 312 TFLOPS（FP16）
```

cuBLAS 自动使用 Tensor Core（当数据类型和矩阵尺寸满足条件时）：
- 数据类型：FP16、BF16、TF32、INT8
- 矩阵尺寸：M, N, K 都是 8 的倍数（最好是 16 或 64 的倍数）

---

## 五、cuBLAS 简单使用

```c
#include <cublas_v2.h>

// cuBLAS SGEMM：单精度矩阵乘法
void cublas_gemm_example(int M, int N, int K) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    float alpha = 1.0f, beta = 0.0f;
    
    // 注意：cuBLAS 使用列主序（Column-Major）！
    // C (M×N) = A (M×K) × B (K×N)
    // 等价的列主序调用：C_cm = B_cm × A_cm
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置
        N, M, K,                    // n, m, k（列主序交换）
        &alpha,
        d_B, N,                     // B 的 leading dimension
        d_A, K,                     // A 的 leading dimension
        &beta,
        d_C, N                      // C 的 leading dimension
    );
    
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
```

---

## 六、PyTorch 中的 cuBLAS

```python
import torch
import time

device = torch.device('cuda')

M, K, N = 4096, 4096, 4096

# FP32 GEMM
A = torch.randn(M, K, device=device, dtype=torch.float32)
B = torch.randn(K, N, device=device, dtype=torch.float32)

# 预热
for _ in range(3):
    C = torch.mm(A, B)
torch.cuda.synchronize()

# 计时
start = time.time()
for _ in range(10):
    C = torch.mm(A, B)
torch.cuda.synchronize()
elapsed = (time.time() - start) / 10

flops = 2 * M * K * N
tflops = flops / elapsed / 1e12
print(f"FP32 GEMM: {elapsed*1000:.2f}ms, {tflops:.2f} TFLOPS")

# FP16 GEMM（使用 Tensor Core）
A_fp16 = A.half()
B_fp16 = B.half()

for _ in range(3):
    C_fp16 = torch.mm(A_fp16, B_fp16)
torch.cuda.synchronize()

start = time.time()
for _ in range(10):
    C_fp16 = torch.mm(A_fp16, B_fp16)
torch.cuda.synchronize()
elapsed_fp16 = (time.time() - start) / 10

tflops_fp16 = flops / elapsed_fp16 / 1e12
print(f"FP16 GEMM: {elapsed_fp16*1000:.2f}ms, {tflops_fp16:.2f} TFLOPS")
print(f"FP16/FP32 加速比: {tflops_fp16/tflops:.1f}x")
```

---

## 今天的关键认识

1. **GEMM 是神经网络最核心的计算**，cuBLAS 把它优化到接近硬件峰值
2. **Tiled GEMM**：用 Shared Memory 缓存分块，减少全局内存访问
3. **Tensor Core**：FP16/BF16 矩阵乘法，性能是 FP32 的 5-10 倍
4. **矩阵尺寸对齐**：M/N/K 是 64 的倍数时，Tensor Core 利用率最高

---

## 明天预告

D2：**cuBLAS API 与性能调优**——不同的 API 参数如何影响性能，如何做自动调优。
