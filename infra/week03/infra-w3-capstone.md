---
layout: default
title: "D7 · 综合实战：cuBLAS GEMM 全流程"
---

# D7 · AI Infra Week 3 综合实战

> **从手写 GEMM 到 cuBLAS，再到 PyTorch 生产优化**

---

## 本周回顾

| 天 | 核心 |
|----|------|
| D1 | cuBLAS 架构：Tiled GEMM + Tensor Core |
| D2 | cuBLAS API：TF32 / Batched GEMM / cublasLt |
| D3 | cuDNN 卷积算法：算法选择 / Winograd / benchmark |
| D4 | Workspace 管理 + 算子融合 + Flash Attention |
| D5 | NHWC vs NCHW：内存布局的 20-80% 性能差异 |
| D6 | Profile → 分析 → 优化完整流程 |

---

## 实战项目：Transformer Inference 优化

目标：把一个基础 Transformer 的推理速度提升 3x+

```python
import torch
import torch.nn as nn
import time
from torch.profiler import profile, ProfilerActivity

# ============================================================
# 基础版本（未优化）
# ============================================================

class BaselineTransformer(nn.Module):
    def __init__(self, vocab_size=32000, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

# 推理 benchmark
def benchmark_inference(model, input_ids, n=100, warmup=10):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(n):
            _ = model(input_ids)
        torch.cuda.synchronize()
    
    return (time.time() - start) / n * 1000  # ms

device = torch.device('cuda')
input_ids = torch.randint(0, 32000, (8, 512), device=device)

# 基础版本（FP32）
model_baseline = BaselineTransformer().to(device)
t_baseline = benchmark_inference(model_baseline, input_ids)
print(f"基础版本 (FP32): {t_baseline:.2f}ms")

# ============================================================
# 优化版本 1：FP16
# ============================================================
model_fp16 = BaselineTransformer().to(device).half()
input_ids_fp16 = input_ids  # embedding 输入仍是 int
t_fp16 = benchmark_inference(model_fp16, input_ids)
print(f"FP16: {t_fp16:.2f}ms ({t_baseline/t_fp16:.2f}x)")

# ============================================================
# 优化版本 2：FP16 + NHWC（这里 Transformer 主要受益于 FP16，NHWC 对 CNN 更明显）
# ============================================================

# ============================================================
# 优化版本 3：torch.compile
# ============================================================
model_compiled = torch.compile(
    BaselineTransformer().to(device).half(),
    mode='max-autotune'
)
# 需要预热（compile 在第一次运行时完成）
for _ in range(5):
    _ = model_compiled(input_ids)
torch.cuda.synchronize()

t_compiled = benchmark_inference(model_compiled, input_ids)
print(f"Compiled (FP16): {t_compiled:.2f}ms ({t_baseline/t_compiled:.2f}x)")

# ============================================================
# 优化版本 4：Flash Attention（对长序列尤其有效）
# ============================================================
class FlashTransformer(nn.Module):
    def __init__(self, vocab_size=32000, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        # PyTorch 2.0+ TransformerEncoderLayer 自动使用 Flash Attention
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, n_heads, batch_first=True,
                norm_first=True  # Pre-LN，更稳定
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True
        ):
            for layer in self.layers:
                x = layer(x)
        return self.head(self.norm(x))

model_flash = FlashTransformer().to(device).half()
t_flash = benchmark_inference(model_flash, input_ids)
print(f"Flash Attention (FP16): {t_flash:.2f}ms ({t_baseline/t_flash:.2f}x)")

# ============================================================
# 优化版本 5：所有技术叠加
# ============================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

model_full = torch.compile(
    FlashTransformer().to(device).half(),
    mode='max-autotune'
)
for _ in range(5):
    _ = model_full(input_ids)
torch.cuda.synchronize()

t_full = benchmark_inference(model_full, input_ids)
print(f"\n全优化版本: {t_full:.2f}ms ({t_baseline/t_full:.2f}x)")
print(f"{'='*40}")
print(f"基础 FP32:        {t_baseline:.2f}ms (1.0x)")
print(f"FP16:             {t_fp16:.2f}ms ({t_baseline/t_fp16:.2f}x)")
print(f"Compiled FP16:    {t_compiled:.2f}ms ({t_baseline/t_compiled:.2f}x)")
print(f"Flash Attn FP16:  {t_flash:.2f}ms ({t_baseline/t_flash:.2f}x)")
print(f"全优化:           {t_full:.2f}ms ({t_baseline/t_full:.2f}x)")
```

---

## Week 3 完成！

🎉 **AI Infra Week 3 全部完成！**

- ✅ cuBLAS GEMM：Tiled GEMM + Tensor Core 原理
- ✅ cuBLAS API：TF32 / Batched GEMM / cublasLt
- ✅ cuDNN 卷积算法：自动选择 + Winograd
- ✅ 算子融合：Flash Attention 原理
- ✅ NHWC 内存布局：20-80% 性能提升
- ✅ Profile + 优化：系统性调优方法论

**Week 4 预告**：CUDA Kernel 编写综合实战——手写 Softmax、LayerNorm，并与 Triton 对比
