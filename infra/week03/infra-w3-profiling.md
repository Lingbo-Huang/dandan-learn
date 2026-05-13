---
layout: default
title: "D6 · 综合调优：Profile → 分析 → 优化"
---

# D6 · 性能调优完整流程

> **AI Infra Week 3**  
> 性能优化的第一原则：先量化，再优化。不 Profile 就动手是浪费时间。

---

## 一、性能调优方法论

```
1. 设立性能目标（要达到多快？）
2. Profile（找到真正的瓶颈）
3. 分析根因（是计算受限？内存受限？通信受限？）
4. 优化（针对根因用对应技术）
5. 验证（确认优化有效且没有破坏精度）
6. 回到 2（迭代优化）
```

---

## 二、PyTorch Profiler 完整使用

```python
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

model = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
    num_layers=6
).cuda().half()

x = torch.randn(8, 512, 512, device='cuda', dtype=torch.float16)

# 完整 Profile（包括内存）
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    on_trace_ready=tensorboard_trace_handler('./log/trace'),  # TensorBoard 可视化
) as prof:
    # 跳过前几步（预热）
    for step in range(10):
        with record_function(f"step_{step}"):
            output = model(x)
        prof.step()

# 按 CUDA 时间排序，显示 Top 10
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=10,
    max_name_column_width=40
))

# 查看内存使用
print(prof.key_averages().table(
    sort_by="cuda_memory_usage",
    row_limit=5
))
```

---

## 三、NVIDIA Nsight Systems（专业工具）

```bash
# 收集 timeline
nsys profile \
  --output=profile_result \
  --trace=cuda,cudnn,cublas \
  --cuda-memory-usage=true \
  python your_script.py

# 收集 kernel 级别性能
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
              l1tex__t_bytes.sum,\
              lts__t_bytes.sum \
    python your_script.py

# 查看结果
nsys-ui profile_result.nsys-rep
```

**关键指标**：

| 指标 | 含义 | 优化方向 |
|------|------|---------|
| SM Utilization | GPU SM 使用率 | <70% → 找 kernel launch 开销 |
| Memory BW | 内存带宽利用率 | 高→内存受限→算子融合 |
| Tensor Core Utilization | Tensor Core 使用率 | 低→检查数据类型、矩阵尺寸 |
| DRAM Read/Write | HBM 读写量 | 高→需要算子融合或更好的缓存 |

---

## 四、常见性能问题与解决方案

```python
import torch
import torch.nn as nn

# 问题1: CPU-GPU 同步（常见坑）
def bad_code(model, data_loader):
    for batch in data_loader:
        output = model(batch.cuda())
        loss_val = float(output.loss)  # 这行会强制 CPU-GPU 同步！
        print(f"Loss: {loss_val}")     # 每步都阻塞，GPU 空转

def good_code(model, data_loader):
    for i, batch in enumerate(data_loader):
        output = model(batch.cuda())
        loss = output.loss  # 保留 Tensor，不转为 Python 标量
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:  # 只偶尔同步
            print(f"Loss: {loss.item()}")  # 这里才同步

# 问题2: DataLoader 是瓶颈
from torch.utils.data import DataLoader

slow_loader = DataLoader(dataset, batch_size=32, num_workers=0)  # 单进程，很慢
fast_loader = DataLoader(
    dataset, 
    batch_size=32, 
    num_workers=8,           # 多进程预加载
    pin_memory=True,         # 使用锁页内存，加速 CPU→GPU 传输
    prefetch_factor=2,       # 提前准备 2 个 batch
    persistent_workers=True  # 避免每 epoch 重建 worker 进程
)

# 问题3: 小 batch size 导致 GPU 利用率低
# 解决：梯度累积
accumulation_steps = 4
for i, batch in enumerate(loader):
    output = model(batch)
    loss = criterion(output) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # 等效于 batch_size × 4 的更新
        optimizer.zero_grad()
```

---

## 五、综合优化 Checklist

```python
# 一个生产环境的优化检查清单

# ✅ 数据类型
model = model.to(torch.float16)  # 或 bfloat16

# ✅ TF32（A100 上）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ✅ cuDNN benchmark
torch.backends.cudnn.benchmark = True  # 固定输入尺寸时

# ✅ NHWC（CNN 模型）
model = model.to(memory_format=torch.channels_last)

# ✅ 混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    loss = model(x)

# ✅ torch.compile（PyTorch 2.0+）
model = torch.compile(model, mode='max-autotune')

# ✅ Flash Attention
with torch.backends.cuda.sdp_kernel(enable_flash=True):
    attn_output = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

# ✅ DataLoader 优化
loader = DataLoader(dataset, num_workers=8, pin_memory=True, persistent_workers=True)

# ✅ 梯度累积（等效大 batch）
# ✅ 避免不必要的 CPU-GPU 同步
```

---

## 今天的关键认识

1. **先 Profile，再优化**：不量化就瞎猜，是工程师最大的浪费
2. **CPU-GPU 同步是常见瓶颈**：`float(tensor)` 会触发同步，尽量减少
3. **DataLoader 经常是瓶颈**：`num_workers`, `pin_memory` 是必须配置的
4. **综合优化**：所有技术组合使用，端到端提速 3-5x 是常见的

---

## 明天预告

D7：**综合实战**——手写 cuBLAS GEMM 并验证，Week 3 完整项目收官。
