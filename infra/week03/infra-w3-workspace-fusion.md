---
layout: default
title: "D4 · Workspace 管理与算子融合"
---

# D4 · Workspace 管理与算子融合

> **AI Infra Week 3**  
> 内存是 GPU 性能的第二大限制（仅次于计算）。算子融合是减少内存访问的核心手段。

---

## 一、内存访问是瓶颈

现代 GPU 的计算峰值（FLOPS）远高于内存带宽的理论极限：

```
A100 SXM:
  FP16 峰值计算：312 TFLOPS
  内存带宽：     2.0 TB/s

如果一个操作的"算术强度"（FLOPS/Byte）太低，
就是内存带宽受限（Memory Bound），计算峰值发挥不出来
```

**典型算子的算术强度**：

| 算子 | 算术强度 | 受限类型 |
|------|---------|---------|
| 大矩阵乘法 | >100 FLOPS/Byte | 计算受限 |
| 向量加法 | ~1 FLOPS/Byte | 内存受限 |
| LayerNorm | ~10 FLOPS/Byte | 内存受限 |
| ReLU | ~0.5 FLOPS/Byte | 严重内存受限 |

---

## 二、算子融合（Operator Fusion）

**问题**：连续的 Elementwise 操作（ReLU、Dropout、Bias Add）每个都读写一次全局内存。

**解决**：把多个操作合并成一个 Kernel，只读写一次。

```
未融合：
  读 X → [LayerNorm] → 写 Y → 读 Y → [GELU] → 写 Z → 读 Z → [Dropout] → 写 W
  内存访问：6次（3读3写）

融合后：
  读 X → [LayerNorm + GELU + Dropout] → 写 W
  内存访问：2次（1读1写）
  
理论加速：3x
```

---

## 三、torch.compile 自动融合

```python
import torch
import torch.nn as nn
import time

class TransformerBlock(nn.Module):
    def __init__(self, dim=1024, num_heads=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

device = torch.device('cuda')
model = TransformerBlock(dim=1024, num_heads=16).to(device).half()

# 使用 torch.compile 自动进行算子融合
compiled_model = torch.compile(model, mode='max-autotune')

x = torch.randn(8, 512, 1024, device=device, dtype=torch.float16)

# 预热
for _ in range(5):
    y = model(x)
    y_compiled = compiled_model(x)
torch.cuda.synchronize()

# 对比
n_iter = 50
start = time.time()
for _ in range(n_iter):
    y = model(x)
torch.cuda.synchronize()
t_eager = (time.time() - start) / n_iter * 1000

start = time.time()
for _ in range(n_iter):
    y_compiled = compiled_model(x)
torch.cuda.synchronize()
t_compiled = (time.time() - start) / n_iter * 1000

print(f"Eager 模式: {t_eager:.2f}ms")
print(f"Compiled 模式: {t_compiled:.2f}ms")
print(f"加速比: {t_eager/t_compiled:.2f}x")
```

---

## 四、Flash Attention：算子融合的极致

标准 Attention 计算：

```python
# 标准实现（多次读写 HBM）
def standard_attention(Q, K, V, scale):
    S = Q @ K.T * scale          # 写 S 到 HBM
    P = torch.softmax(S, dim=-1) # 读 S, 写 P 到 HBM
    O = P @ V                    # 读 P, V, 写 O 到 HBM
    return O
    # HBM 访问：O(n²) 大小的矩阵读写 4 次
```

Flash Attention 的核心思想：

```python
# Flash Attention 伪代码（实际用 CUDA 实现）
# 关键：分块计算，每块保持在 SRAM（共享内存）里，不写回 HBM
def flash_attention(Q, K, V, block_size=64):
    # 按 block_size 分块
    for i in range(0, seq_len, block_size):
        for j in range(0, seq_len, block_size):
            Q_block = Q[i:i+block_size]  # 从 HBM 读一次
            K_block = K[j:j+block_size]
            V_block = V[j:j+block_size]
            
            # 在 SRAM 里完成这个分块的 softmax 和 V 加权
            S_block = Q_block @ K_block.T
            # ... online softmax 算法 ...
            
            # 累积结果，不写中间结果到 HBM
    
    # 只写最终结果到 HBM
    # HBM 访问：O(n²) → O(n)，显存节省 4-8 倍！
```

```python
# 在 PyTorch 中使用 Flash Attention
import torch
from torch.nn.functional import scaled_dot_product_attention

B, H, S, D = 8, 16, 2048, 64
Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

# PyTorch 2.0+ 自动使用 Flash Attention（当条件满足时）
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
    output = scaled_dot_product_attention(Q, K, V, scale=1.0/D**0.5)

print(f"输出形状: {output.shape}")
print(f"已自动使用 Flash Attention！")
```

---

## 五、Workspace 预分配策略

```python
import torch

# 坏的做法：每次推理都分配/释放内存（有延迟）
def slow_inference(model, x):
    return model(x)  # 内部动态分配 workspace

# 好的做法：预分配 workspace，推理时直接复用
class OptimizedInference:
    def __init__(self, model, input_shape):
        self.model = model
        # 预热一次，让 cuDNN 缓存算法选择
        dummy_input = torch.zeros(input_shape, device='cuda', dtype=torch.float16)
        with torch.no_grad():
            _ = self.model(dummy_input)
        torch.cuda.synchronize()
        
        # 预分配输出缓冲区
        output_shape = self.model(dummy_input).shape
        self.output_buffer = torch.empty(output_shape, device='cuda', dtype=torch.float16)
    
    @torch.no_grad()
    def infer(self, x):
        return self.model(x)
```

---

## 今天的关键认识

1. **算子融合**：把多个 Elementwise 操作合并，减少内存读写次数
2. **torch.compile**：自动进行算子融合（Dynamo + Inductor）
3. **Flash Attention**：算子融合的极致，把 O(n²) 内存降到 O(n)
4. **Workspace 预分配**：避免推理时动态分配的延迟

---

## 明天预告

D5：**NHWC vs NCHW 内存布局**——数据排列方式对性能的巨大影响。
