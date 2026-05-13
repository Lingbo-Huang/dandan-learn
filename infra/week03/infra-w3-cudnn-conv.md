---
layout: default
title: "D3 · cuDNN 卷积算法选择"
---

# D3 · cuDNN 卷积算法选择

> **AI Infra Week 3**  
> 同一个卷积操作有多种计算方法，选对了能差 5 倍的性能。cuDNN 帮你选最优算法。

---

## 一、卷积的多种实现算法

| 算法 | 原理 | 适用场景 |
|------|------|---------|
| **Direct Conv** | 直接按定义计算 | 小卷积核（1x1, 3x3），稠密数据 |
| **im2col + GEMM** | 展开输入，转为矩阵乘法 | 通用场景，易于优化 |
| **FFT** | 频域计算 | 大卷积核（≥7x7） |
| **Winograd** | 减少乘法次数的算法 | 3x3, 5x5 卷积，特定 batch size |

**cuDNN 根据参数自动选择**：

- 卷积核大小、stride、padding
- 输入/输出尺寸
- 数据类型（FP32/FP16）
- 可用显存大小

---

## 二、cuDNN Benchmark 模式

```python
import torch
import torch.nn as nn
import time

# cudnn.benchmark = True：第一次运行时测试多种算法，选最快的
# 适合：输入尺寸固定的场景（如固定分辨率的图像分类）
torch.backends.cudnn.benchmark = True

# 验证：benchmark 对固定尺寸的影响
device = torch.device('cuda')
model = nn.Sequential(
    nn.Conv2d(64, 256, 3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
).to(device)

input_tensor = torch.randn(32, 64, 56, 56, device=device)

# 第一次（选择算法）
start = time.time()
output = model(input_tensor)
torch.cuda.synchronize()
print(f"首次运行（含算法选择）: {(time.time()-start)*1000:.2f}ms")

# 后续（直接用最优算法）
times = []
for _ in range(20):
    start = time.time()
    output = model(input_tensor)
    torch.cuda.synchronize()
    times.append((time.time()-start)*1000)

print(f"后续平均: {sum(times)/len(times):.2f}ms")
```

---

## 三、cuDNN API 手动选择算法

```c
#include <cudnn.h>

void cudnn_conv_example() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);
    
    // 描述符：输入、滤波器、输出
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);
    
    // 设置输入：batch=32, channel=64, h=56, w=56，FP16
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NHWC,
                               CUDNN_DATA_HALF, 32, 64, 56, 56);
    
    // 设置卷积滤波器：256个滤波器，64通道，3×3
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_HALF,
                               CUDNN_TENSOR_NHWC, 256, 64, 3, 3);
    
    // 设置卷积参数：padding=1, stride=1, dilation=1
    cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1,
                                   CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    
    // 启用 Tensor Core
    cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);
    
    // 自动找最优算法
    int num_algos;
    cudnnConvolutionFwdAlgoPerf_t algo_perf[10];
    
    cudnnFindConvolutionForwardAlgorithm(
        handle, input_desc, filter_desc, conv_desc, output_desc,
        10, &num_algos, algo_perf
    );
    
    // 使用最优算法（algo_perf[0]）
    printf("最优算法: %d, 时间: %.3f ms, 内存: %zu MB\n",
           algo_perf[0].algo,
           algo_perf[0].time,
           algo_perf[0].memory / 1024 / 1024);
    
    // 执行卷积
    size_t workspace_size = algo_perf[0].memory;
    void *workspace;
    cudaMalloc(&workspace, workspace_size);
    
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(
        handle, &alpha,
        input_desc, d_input,
        filter_desc, d_filter,
        conv_desc, algo_perf[0].algo,
        workspace, workspace_size,
        &beta,
        output_desc, d_output
    );
}
```

---

## 四、不同算法的 Workspace 需求

```python
# cuDNN 的不同算法需要不同大小的临时内存（workspace）
# workspace 越大，通常有越好的算法可选

import torch
import torch.nn as nn

# 查看卷积的 workspace 使用
torch.backends.cudnn.benchmark = True

conv = nn.Conv2d(64, 256, 3, padding=1).cuda().half()
x = torch.randn(32, 64, 56, 56, device='cuda', dtype=torch.float16)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    profile_memory=True
) as prof:
    y = conv(x)

print(prof.key_averages().table(sort_by="cuda_memory_usage"))
```

---

## 五、Winograd 算法深入

对 3×3 卷积，Winograd F(2,3) 算法将乘法次数从 36 减少到 16（4×加速）：

```
标准卷积（3×3 → 2×2 输出）：3×3×3×3 = 36 次乘法
Winograd F(2,3)：                     16 次乘法（减少了 2.25×）
```

**使用限制**：
- 只适合特定卷积尺寸（3×3, 5×5）
- 需要额外的变换步骤（有开销）
- batch size 小时可能不划算
- FP16 时精度问题，有时需要关闭

```python
# 禁用 Winograd（当精度有问题时）
import os
os.environ['CUDNN_DISABLE_CONV_NCHW_WINOGRAD'] = '1'
# 或者通过 cudnn flags
torch.backends.cudnn.allow_tf32 = False  # 更保守的精度设置
```

---

## 今天的关键认识

1. **cuDNN 有多种卷积算法**：Direct/im2col/FFT/Winograd，各有适用场景
2. **benchmark=True**：对固定尺寸的输入，自动找最优算法（推理场景的默认选择）
3. **Workspace**：更大的 Workspace 通常意味着更快的算法
4. **Winograd**：3×3 卷积神器，但有精度和场景限制

---

## 明天预告

D4：**cuDNN Workspace 管理与算子融合**——内存优化的核心技术，让推理速度再提升。
