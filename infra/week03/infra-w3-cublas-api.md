---
layout: default
title: "D2 · cuBLAS API 与性能调优"
---

# D2 · cuBLAS API 与性能调优

> **AI Infra Week 3**  
> cuBLAS 的 API 设计、math mode 选择、批量 GEMM，以及如何用 cublasLt 做自动调优。

---

## 一、cublasHandle 和 Math Mode

```c
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

// 设置 Math Mode：控制是否使用 Tensor Core
cublasMath_t math_mode;

// 默认：可能用 Tensor Core（取决于数据类型）
cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

// 强制 Tensor Core（允许精度损失）
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

// TF32 模式：FP32 输入，但用更低精度的 Tensor Core 运算
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
```

**TF32 精度说明**：
- FP32：1位符号 + 8位指数 + 23位尾数
- TF32：1位符号 + 8位指数 + **10位尾数**（截断）
- 精度：约等于 FP16，范围等于 FP32
- 速度：比 FP32 快约 8-10 倍（Tensor Core）
- PyTorch 中通过 `torch.backends.cuda.matmul.allow_tf32 = True` 开启

---

## 二、批量 GEMM（Batched GEMM）

Transformer 中，多头注意力需要同时做多个矩阵乘法：

```c
// 批量 GEMM：同时计算 batch_size 个矩阵乘法
cublasSgemmBatched(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K,
    &alpha,
    d_B_array, N,   // 指针数组，每个指向一个 B 矩阵
    d_A_array, K,   // 指针数组，每个指向一个 A 矩阵
    &beta,
    d_C_array, N,   // 指针数组，每个指向一个 C 矩阵
    batch_size      // batch 数量
);

// Strided 版本（连续内存，更常用）
cublasSgemmStridedBatched(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K,
    &alpha,
    d_B, N,                // B 的起始地址
    (long long)K * N,      // 相邻 batch 的 B 的跨度（stride）
    d_A, K,
    (long long)M * K,
    &beta,
    d_C, N,
    (long long)M * N,
    batch_size
);
```

**PyTorch 中的 Batched GEMM**：

```python
import torch

# Multi-head attention 中的批量矩阵乘法
batch_size, num_heads, seq_len, head_dim = 8, 16, 512, 64

Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)

# 底层调用 cublasSgemmStridedBatched
attn_weights = torch.bmm(
    Q.view(-1, seq_len, head_dim),
    K.view(-1, seq_len, head_dim).transpose(-2, -1)
) / (head_dim ** 0.5)

print(f"输入: {Q.shape}")
print(f"输出: {attn_weights.shape}")
print(f"Batch size: {batch_size * num_heads}")  # 128 个并行 GEMM
```

---

## 三、cublasLt：灵活的矩阵乘法

cublasLt（cuBLAS Lightweight）提供更细粒度的控制：

```c
#include <cublasLt.h>

cublasLtHandle_t lt_handle;
cublasLtCreate(&lt_handle);

// 矩阵布局描述符
cublasLtMatrixLayout_t matA, matB, matC;
cublasLtMatrixLayoutCreate(&matA, CUDA_R_16F, M, K, M);  // FP16 列主序
cublasLtMatrixLayoutCreate(&matB, CUDA_R_16F, K, N, K);
cublasLtMatrixLayoutCreate(&matC, CUDA_R_16F, M, N, M);

// 矩阵乘法描述符
cublasLtMatmulDesc_t matmul_desc;
cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);  // 计算用 FP32，IO 用 FP16

// 自动调优：找到最快的算法
cublasLtMatmulPreference_t preference;
cublasLtMatmulPreferenceCreate(&preference);
// 设置 workspace 大小限制（更大的 workspace 通常有更好的算法可选）
size_t workspace_size = 128 * 1024 * 1024;  // 128 MB
cublasLtMatmulPreferenceSetAttribute(
    preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
    &workspace_size, sizeof(workspace_size)
);

// 查询最优算法
int num_results;
cublasLtMatmulHeuristicResult_t result;
cublasLtMatmulAlgoGetHeuristic(
    lt_handle, matmul_desc,
    matA, matB, matC, matC,
    preference, 1, &result, &num_results
);

// 使用最优算法执行
void *workspace;
cudaMalloc(&workspace, workspace_size);
cublasLtMatmul(
    lt_handle, matmul_desc,
    &alpha, d_A, matA, d_B, matB,
    &beta,  d_C, matC, d_C, matC,
    &result.algo,
    workspace, workspace_size,
    0  // stream
);
```

---

## 四、性能调优检查清单

```python
# PyTorch 中的 cuBLAS 优化设置
import torch

# 1. 允许 TF32（A100 上 FP32 计算的 8-10x 提速，精度略降）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 2. 使用 FP16/BF16（Tensor Core 加速）
model = model.half()  # FP16
# 或
model = model.to(torch.bfloat16)  # BF16（范围更大，适合训练）

# 3. 矩阵尺寸对齐（关键！）
# 确保 batch_size, hidden_dim 等是 64（最好是 256）的倍数
# 不对齐的尺寸会有 padding overhead

# 4. 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # 防止 FP16 梯度下溢

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():  # 自动将部分运算转为 FP16
        output = model(batch)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()   # 梯度缩放
    scaler.step(optimizer)
    scaler.update()

# 5. cuDNN benchmark（自动选最优卷积算法）
torch.backends.cudnn.benchmark = True
```

---

## 五、性能 Profile：找到真正的瓶颈

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = ...  # 你的模型
input_data = ...

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    with record_function("model_inference"):
        output = model(input_data)

# 查看最耗时的操作
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=10
))
# 输出：每个操作的 CPU/CUDA 时间，帮助定位瓶颈
```

---

## 今天的关键认识

1. **TF32 模式**：精度略低但快 8-10 倍，训练中几乎没有影响，默认开启
2. **Batched GEMM**：Transformer 多头注意力的关键，128 个并行 GEMM 同时跑
3. **cublasLt 自动调优**：给定 workspace，自动找最优算法
4. **矩阵尺寸对齐**：64 的倍数是红线，256 的倍数效率最优

---

## 明天预告

D3：**cuDNN 卷积算法选择**——卷积有多种实现算法，如何让 cuDNN 自动选择最优的？
