---
layout: default
title: "D5 · NHWC vs NCHW：内存布局"
---

# D5 · NHWC vs NCHW：内存布局的性能影响

> **AI Infra Week 3**  
> 同样的数据，不同的排列方式，性能可以差 2 倍。内存布局是性能优化里最容易被忽视的部分。

---

## 一、两种布局

对于形状为 `[N, C, H, W]` 的 4D 张量（batch, channel, height, width）：

**NCHW（Channel First，PyTorch 默认）**：
```
在内存中：[N0,C0,H0,W0], [N0,C0,H0,W1], ..., [N0,C0,H1,W0], ..., [N0,C1,H0,W0], ...
特征：同一 Channel 的相邻像素在内存中相邻
```

**NHWC（Channel Last）**：
```
在内存中：[N0,H0,W0,C0], [N0,H0,W0,C1], ..., [N0,H0,W0,C_last], [N0,H0,W1,C0], ...
特征：同一像素的所有 Channel 在内存中相邻
```

---

## 二、为什么布局影响性能？

**卷积的访问模式**：

```
NCHW（不利于 Tensor Core）：
  卷积核滑动时，访问多个 channel 的同一位置
  → 内存访问不连续，cache miss 多

NHWC（有利于 Tensor Core）：
  访问同一位置的所有 channel 值（SIMD 友好）
  → 内存访问连续，Tensor Core 利用率高
```

**实测结果（A100，FP16）**：

| 操作 | NCHW | NHWC | 加速比 |
|------|------|------|--------|
| Conv2d 3×3 | 1.0x | 1.3-1.8x | 30-80% |
| BatchNorm | 1.0x | 1.5x | 50% |
| 端到端 ResNet | 1.0x | 1.2x | 20% |

---

## 三、PyTorch 中使用 NHWC

```python
import torch
import torch.nn as nn
import time

# 创建 NHWC 格式的张量
x_nchw = torch.randn(32, 64, 56, 56, device='cuda', dtype=torch.float16)
x_nhwc = x_nchw.to(memory_format=torch.channels_last)

print(f"NCHW shape: {x_nchw.shape}, stride: {x_nchw.stride()}")
print(f"NHWC shape: {x_nhwc.shape}, stride: {x_nhwc.stride()}")
# stride 不同，但 shape 相同
# NCHW stride: (200704, 3136, 56, 1)  — C 方向步长大
# NHWC stride: (200704, 1, 3584, 64) — C 方向步长小（连续）

# 将模型转为 NHWC
model = nn.Sequential(
    nn.Conv2d(64, 256, 3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.Conv2d(256, 128, 1),
).to('cuda').half()

model_nhwc = model.to(memory_format=torch.channels_last)

# 性能对比
def benchmark(model, x, n=50):
    for _ in range(5):  # 预热
        _ = model(x)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(n):
        _ = model(x)
    torch.cuda.synchronize()
    return (time.time() - start) / n * 1000

t_nchw = benchmark(model, x_nchw)
t_nhwc = benchmark(model_nhwc, x_nhwc)

print(f"\nNCHW: {t_nchw:.2f}ms")
print(f"NHWC: {t_nhwc:.2f}ms")
print(f"NHWC 加速: {t_nchw/t_nhwc:.2f}x")
```

---

## 四、整个模型转为 NHWC

```python
# 推荐做法：创建时就指定 NHWC
model = ResNet50().cuda().half().to(memory_format=torch.channels_last)

# 输入也要转换（否则每次推理都会有 layout 转换开销）
x = x.to(memory_format=torch.channels_last)

output = model(x)  # 全程 NHWC，无额外转换
```

---

## 五、布局对 Transformer 的影响

**Transformer 中的 GEMM 是列主序还是行主序？**

```python
import torch

# PyTorch 的矩阵是行主序（Row-Major）
A = torch.randn(512, 768)
print(f"Stride: {A.stride()}")  # (768, 1)：行方向步长768，列方向步长1

# cuBLAS 内部是列主序，PyTorch 在调用时会做等价转换
# 实际上 PyTorch 的 mm/bmm 已经做了这个透明处理

# 关键：确保张量是连续的（contiguous），否则 cuBLAS 无法高效处理
x = A.T  # 转置后不连续
print(f"Contiguous: {x.is_contiguous()}")  # False

x_cont = x.contiguous()  # 强制变为连续
print(f"Contiguous after: {x_cont.is_contiguous()}")  # True
```

---

## 今天的关键认识

1. **NHWC 在现代 GPU 上通常比 NCHW 更快**，因为 Tensor Core 更喜欢这种访问模式
2. **PyTorch 支持 NHWC**：`tensor.to(memory_format=torch.channels_last)`
3. **输入和模型都要转换**，否则会有 layout 转换开销
4. **保证 contiguous**：GEMM 操作要求连续内存，转置后记得调 `.contiguous()`

---

## 明天预告

D6：**综合调优实战**——Profile → 找瓶颈 → 优化的完整流程。
