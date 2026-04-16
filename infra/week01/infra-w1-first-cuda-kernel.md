# D4：第一个 CUDA Kernel——矩阵转置与合并访问

> Week1 主题：GPU架构与CUDA基础 | AI Infra线

---

## 🎯 学习目标

- 独立编写、编译并运行一个完整的 CUDA Kernel
- 通过矩阵转置理解合并内存访问 vs 非合并访问的性能差异
- 学会使用共享内存解决 Bank Conflict 问题
- 使用 CUDA Event 进行性能计时

---

## 🧠 核心知识点

### 1. 矩阵转置——一个绝佳的教学案例

矩阵转置：将矩阵 A[M][N] 转置为 B[N][M]，即 `B[j][i] = A[i][j]`

这个操作的挑战：**读和写不能同时都是合并访问**

```
A (行主序):          B = A^T (行主序):
┌─────────────┐      ┌─────────────┐
│ 0  1  2  3  │      │ 0  4  8 12  │
│ 4  5  6  7  │  →   │ 1  5  9 13  │
│ 8  9 10 11  │      │ 2  6 10 14  │
│12 13 14 15  │      │ 3  7 11 15  │
└─────────────┘      └─────────────┘

方案一（Naive）：让线程 (i,j) 完成 B[j][i] = A[i][j]
  - 读 A[i][j]：threadIdx.x 对应 j 方向 → 不同线程读同一行 → ✅ 合并读
  - 写 B[j][i]：threadIdx.x 对应 i 方向 → 不同线程写同一列 → ❌ 非合并写

方案二（另一个 Naive）：让线程 (i,j) 完成 B[i][j] = A[j][i]
  - 读 A[j][i]：threadIdx.x 对应 i → 不合并读 ❌
  - 写 B[i][j]：threadIdx.x 对应 j → 合并写 ✅

结论：转置必然有一端不合并，解决方案是**共享内存做缓冲**
```

### 2. 朴素转置 Kernel（非合并写）

```c
__global__ void transpose_naive(float* in, float* out, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 对应输入列
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 对应输入行
    
    if (row < rows && col < cols) {
        // 读：in[row][col] → 合并读（同行线程读连续地址）✅
        // 写：out[col][rows + row] → 非合并写（同行线程写不连续地址）❌
        out[col * rows + row] = in[row * cols + col];
    }
}
```

### 3. 使用共享内存优化（合并读+合并写）

核心思路：
1. 用 Tile（块）合并读入全局内存 → 写到共享内存（快）
2. 从共享内存转置读出 → 合并写回全局内存

```c
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_shared(float* in, float* out, int rows, int cols) {
    // 静态分配共享内存，+1 避免 Bank Conflict
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;  // 输入列
    int y = blockIdx.y * TILE_DIM + threadIdx.y;  // 输入行
    
    // Phase 1: 合并读全局内存 → 写共享内存
    // 每个线程读 TILE_DIM/BLOCK_ROWS 个元素（处理TILE_DIM×TILE_DIM的块）
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < rows && x < cols) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * cols + x];
        }
    }
    
    __syncthreads();  // 等待共享内存填充完成
    
    // Phase 2: 从共享内存转置读 → 合并写全局内存
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // 转置后的列 = 原来的行block偏移
    y = blockIdx.x * TILE_DIM + threadIdx.y;  // 转置后的行 = 原来的列block偏移
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < cols && x < rows) {
            // 注意：从 tile 读时，行列互换（实现转置）
            out[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
```

### 4. Bank Conflict（共享内存 Bank 冲突）

共享内存被分为 32 个 Bank（每个 Bank 4 字节宽）：
- Bank i 存储地址：`addr[i], addr[i+32], addr[i+64]...`（每4字节一个bank）
- 同一 Warp 内的线程访问**不同 Bank** → 并行，无冲突 ✅
- 同一 Warp 内的线程访问**同一 Bank 的不同地址** → 串行化，Bank Conflict ❌
- 同一 Warp 内的线程访问**完全相同的地址** → Broadcast，无冲突 ✅

```
访问 tile[threadIdx.y][threadIdx.x]（32×32 矩阵）：
  Warp 内各线程 threadIdx.x 不同 → 访问不同列 → 不同 Bank → ✅ 无冲突

但转置后，访问 tile[threadIdx.x][threadIdx.y]：
  Warp 内各线程 threadIdx.x 相同（同一Warp的y不同），但：
  实际上在矩阵列转置时，会有列方向访问 → 可能冲突

解决方案：将 tile 声明为 [TILE_DIM][TILE_DIM + 1]
  - 每行多一列（padding），打破 Bank 对齐
  - tile[0][0]~[0][32] 分布在 Bank 0~32，没有周期冲突
```

### 5. CUDA 性能分析基础

**有效带宽计算**：

```
理论有效带宽 = (读取字节数 + 写入字节数) / 耗时

矩阵转置（M×N，FP32）：
  读取：M × N × 4 字节
  写入：M × N × 4 字节
  理论有效带宽 = 2 × M × N × 4 / time_ms / 1e6  [GB/s]
```

**性能对比指标**：
- GPU 峰值带宽（H100: 3.35 TB/s，RTX 3090: 936 GB/s）
- 达到峰值带宽的百分比

---

## 🖼️ 原理图解

### 共享内存 Tiling 策略

```
输入矩阵（全局内存）：
┌────────────────────────────────┐
│      Tile(0,0)  │ Tile(1,0)   │
│  32列×32行      │             │
├─────────────────┼─────────────┤
│      Tile(0,1)  │ Tile(1,1)   │
└────────────────────────────────┘

每个 Block 处理一个 Tile：
  Phase 1：合并读全局内存的 Tile → 共享内存
  Phase 2：转置读共享内存 → 合并写全局内存的对应 Tile

内存访问模式：
  Phase 1 读：线程(tx, ty) 读 in[y+ty][x+tx] → 行连续 → 合并 ✅
  Phase 2 写：线程(tx, ty) 写 out[y+tx][x+ty] → 行连续 → 合并 ✅
  中间在共享内存完成行列互换（低代价）
```

### Bank Conflict 可视化

```
tile[32][32]（无 padding）：
  Bank: 0  1  2  3  4  ...  31   0  1  2  ...
  addr: [0][1][2][3][4]...[31] [32][33]...

tile[i][j] → Bank = j % 32

访问列 0（tile[0][0], tile[1][0], ..., tile[31][0]）：
  所有元素都在 Bank 0 → 32路 Bank Conflict ❌

tile[32][33]（有 padding）：
  tile[i][j] → Bank = (i * 33 + j) % 32
  列方向不再产生冲突 ✅
```

---

## 🛠️ 动手练习

### 完整代码：矩阵转置三版本对比

```c
// transpose.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// Version 1: Copy（对照组，两端均合并）
__global__ void copy_matrix(float* in, float* out, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (y+j < rows && x < cols) tile[threadIdx.y+j][threadIdx.x] = in[(y+j)*cols+x];
    }
    __syncthreads();
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (y+j < rows && x < cols) out[(y+j)*cols+x] = tile[threadIdx.y+j][threadIdx.x];
    }
}

// Version 2: Naive 转置（合并读，非合并写）
__global__ void transpose_naive(float* in, float* out, int rows, int cols) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (y+j < rows && x < cols) {
            out[x * rows + (y+j)] = in[(y+j) * cols + x];
        }
    }
}

// Version 3: 共享内存优化（两端合并，含 padding 避免 Bank Conflict）
__global__ void transpose_shared(float* in, float* out, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 避免 Bank Conflict
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (y+j < rows && x < cols)
            tile[threadIdx.y+j][threadIdx.x] = in[(y+j)*cols+x];
    }
    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (y+j < cols && x < rows)
            out[(y+j)*rows+x] = tile[threadIdx.x][threadIdx.y+j];
    }
}

float benchmark(void (*kernel)(float*, float*, int, int), float* d_in, float* d_out,
                int rows, int cols, int repeats) {
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    // Warmup
    for (int i = 0; i < 3; i++) kernel<<<grid, block>>>(d_in, d_out, rows, cols);
    
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) kernel<<<grid, block>>>(d_in, d_out, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / repeats;
}

int main() {
    int rows = 4096, cols = 4096;
    size_t bytes = rows * cols * sizeof(float);
    
    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    for (int i = 0; i < rows * cols; i++) h_in[i] = (float)i;
    
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    
    int repeats = 100;
    float gb = 2.0f * bytes / 1e9;
    
    float ms_copy  = benchmark(copy_matrix,     d_in, d_out, rows, cols, repeats);
    float ms_naive = benchmark(transpose_naive,  d_in, d_out, rows, cols, repeats);
    float ms_share = benchmark(transpose_shared, d_in, d_out, rows, cols, repeats);
    
    printf("版本          | 耗时(ms) | 有效带宽(GB/s)\n");
    printf("-------------|----------|---------------\n");
    printf("Copy（对照）  | %8.3f | %.1f\n", ms_copy,  gb/ms_copy*1000);
    printf("Naive 转置    | %8.3f | %.1f\n", ms_naive, gb/ms_naive*1000);
    printf("共享内存优化  | %8.3f | %.1f\n", ms_share, gb/ms_share*1000);
    
    // 验证正确性
    CHECK_CUDA(cudaMemset(d_out, 0, bytes));
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);
    transpose_shared<<<grid, block>>>(d_in, d_out, rows, cols);
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    
    int errors = 0;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (fabs(h_out[j * rows + i] - h_in[i * cols + j]) > 1e-5) errors++;
    printf("\n正确性验证: %d 个错误（应为0）\n", errors);
    
    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
```

```bash
nvcc -O2 -o transpose transpose.cu && ./transpose
```

**预期结果（RTX 3090 大约）**：
```
版本          | 耗时(ms) | 有效带宽(GB/s)
-------------|----------|---------------
Copy（对照）  |    0.9   | 280+
Naive 转置    |    3.5   |  70
共享内存优化  |    1.0   | 250+
```

### 练习 2：思考与扩展

1. 去掉 `[TILE_DIM + 1]` 中的 `+1`，观察性能变化（Bank Conflict 的代价）
2. 尝试修改 `TILE_DIM` 为 16 或 64，观察影响
3. 使用 Nsight Systems 可视化 kernel 执行时间线：
   ```bash
   nsys profile ./transpose
   nsys-ui report1.nsys-rep  # 图形化查看
   ```

---

## 📝 小结

| 技术 | 作用 | 场景 |
|------|------|------|
| 合并内存访问 | 最大化内存带宽利用率 | 所有访问全局内存的场景 |
| 共享内存 Tiling | 将非合并访问转化为合并访问 | 矩阵运算、卷积等 |
| Bank Conflict 消除（padding） | 避免共享内存访问串行化 | 转置、归约等有列访问的场景 |
| CUDA Event 计时 | 精确测量 kernel 耗时 | 性能调优必用 |
| 正确性验证 | 与 CPU 结果对比 | 所有 kernel 开发 |

**关键认知**：在 GPU 上，"正确"和"高效"是两个独立的问题。Naive 实现往往正确，但可能只达到 10% 的峰值性能。优化的第一步永远是：**理解内存访问模式，消除非合并访问**。

---

**明日预告**：D5 将深入 CUDA 内存层次——共享内存的更多用法、归约（Reduction）算法的逐步优化，这是理解 cuDNN 算子优化的基础。
