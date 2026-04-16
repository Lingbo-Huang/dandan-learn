# D6：CUDA 性能分析工具——Nsight Systems & Nsight Compute 实战

> Week1 主题：GPU架构与CUDA基础 | AI Infra线

---

## 🎯 学习目标

- 掌握 Nsight Systems（nsys）：系统级性能时间线分析
- 掌握 Nsight Compute（ncu）：Kernel 级深度性能剖析
- 能读懂关键性能指标：SM Utilization、Memory Throughput、Roofline
- 建立"先 profile 再优化"的正确工程习惯

---

## 🧠 核心知识点

### 1. 性能分析工具概览

| 工具 | 层次 | 主要用途 | 开销 |
|------|------|---------|------|
| `nvidia-smi` | 系统级 | 实时 GPU 利用率、显存、功耗 | 极低 |
| `nvprof`（旧） | Kernel 级 | 老版 CUDA 的 profile 工具 | 中 |
| **Nsight Systems** | 系统级 | CPU+GPU 时间线、CUDA API、内存传输 | 低 |
| **Nsight Compute** | Kernel 级 | 单个 Kernel 的详细硬件指标 | 高 |
| `torch.profiler` | 框架级 | PyTorch 算子耗时、内存分配 | 低~中 |

### 2. Nsight Systems（nsys）——系统时间线

**安装**：随 CUDA Toolkit 附带，或 `apt install nsight-systems`

**核心功能**：
- CPU 线程时间线
- CUDA Kernel 时间线（在哪个 Stream 运行、持续多久）
- CUDA API 调用（cudaMalloc、cudaMemcpy 等）
- PCIe 数据传输
- NVLink 通信（多 GPU）

```bash
# 基本用法
nsys profile --output=report ./my_program

# 常用参数
nsys profile \
  --trace=cuda,nvtx,osrt \     # 追踪 CUDA API, NVTX标注, OS运行时
  --output=report \
  --force-overwrite=true \
  ./my_program args

# 查看报告（命令行摘要）
nsys stats report.nsys-rep

# 图形化查看（需要 Nsight Systems GUI）
nsys-ui report.nsys-rep
```

**关键关注点**：
1. **Kernel 之间是否有空隙？** → 可能有同步等待或 CPU 瓶颈
2. **cudaMemcpy 占比多少？** → 减少不必要的 H↔D 数据传输
3. **是否有 Stream 并发？** → 用多 Stream 隐藏传输延迟

### 3. NVTX 标注（给代码打时间戳）

在代码中插入 NVTX 标注，让 Profile 结果更易读：

```c
// C/CUDA
#include <nvtx3/nvToolsExt.h>

nvtxRangePush("Data Loading");
load_data();
nvtxRangePop();

nvtxRangePush("Forward Pass");
forward_pass();
nvtxRangePop();
```

```python
# Python / PyTorch
import torch
from torch.cuda import nvtx

nvtx.range_push("Encoder Layer")
output = encoder(input)
nvtx.range_pop()

# 或使用装饰器
with torch.cuda.nvtx.range("Attention"):
    attn_out = attention(q, k, v)
```

### 4. Nsight Compute（ncu）——Kernel 深度剖析

**核心功能**：
- Memory 分析：L1/L2/HBM 命中率、带宽利用率
- Compute 分析：SM Utilization、Warp 效率
- Roofline 分析：确定 Kernel 是 Compute-bound 还是 Memory-bound
- Source-level 分析：哪行代码最慢

```bash
# 分析所有 Kernel（基本信息）
ncu ./my_program

# 分析特定 Kernel
ncu --kernel-name "transpose_shared" ./my_program

# 收集完整指标（较慢）
ncu --set full ./my_program

# 收集特定指标组
ncu --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section Roofline \
    ./my_program

# 生成报告文件（可用 ncu-ui 打开）
ncu --output profile_report ./my_program
ncu-ui profile_report.ncu-rep
```

### 5. 关键性能指标解读

#### 5.1 SM Utilization（SM 利用率）

```
理论 SM 利用率：活跃 SM 数 / 总 SM 数
实际计算利用率（Compute Throughput）：实际 FLOPS / 峰值 FLOPS

如果 SM Util = 30%：
  → Grid 太小（Block 数 < SM 数）
  → 增大 Grid 或处理更大的数据
```

#### 5.2 Memory Throughput（内存吞吐量）

```
L1 Cache Hit Rate：高 → 好（减少 L2/HBM 压力）
L2 Cache Hit Rate：高 → 比直接 HBM 快约 10x
Global Memory Throughput：接近峰值带宽 → 内存访问效率高

关键：L1 Hit Rate 低 + 高 HBM 带宽占用 → 内存访问模式需优化（合并访问）
```

#### 5.3 Warp Efficiency 相关指标

```
Warp Execution Efficiency：
  = 每次指令平均活跃线程数 / 32（Warp大小）
  
100% → 无 Warp Divergence 或边界问题
50%  → 大量分叉或无效线程

Achieved Occupancy：实际 Warp 占用率（越高 → 越能隐藏延迟）
Theoretical Occupancy：理论最大 Warp 占用率
```

#### 5.4 Roofline 模型结果解读

```
ncu 输出的 Roofline 位置：

点在"Memory Roof"左侧（低算术强度区域）→ Memory-bound Kernel
  优化方向：减少内存访问（更多缓存复用、合并访问）
  
点在"Compute Roof"下方（高算术强度区域）→ Compute-bound Kernel  
  优化方向：使用 Tensor Core（FP16、BF16、TF32）、提高 Occupancy

点接近两条线的交叉点 → 平衡点，接近理论峰值 → 已经很好了！
```

### 6. PyTorch Profiler 使用

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = MyModel().cuda()
inputs = torch.randn(batch_size, seq_len, hidden).cuda()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("model_inference"):
        for _ in range(10):
            output = model(inputs)
            torch.cuda.synchronize()

# 打印 Top K 耗时算子
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# 导出 Chrome Trace（用 chrome://tracing/ 查看）
prof.export_chrome_trace("trace.json")

# 导出 TensorBoard（用 tensorboard --logdir=tb_trace 查看）
prof.export_stacks("stacks.txt", metric="self_cuda_time_total")
```

### 7. nvidia-smi 实时监控

```bash
# 实时监控 GPU 利用率（每秒刷新）
nvidia-smi dmon -s u -d 1

# 监控所有指标
nvidia-smi dmon -d 1

# 查看 GPU 功耗、温度
nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 1

# 查看 GPU 上运行的进程
nvidia-smi pmon -d 1
```

---

## 🖼️ 原理图解

### Nsight Systems 时间线示意

```
时间轴 →

CPU Thread 1: [  cudaMalloc  ][cudaMemcpy H→D][  launch kernel  ][cudaMemcpy D→H]
                                    |                   |
CUDA Stream 0:                [H→D Transfer]       [kernel exec]
                                                         |
GPU SM:                                            [WWWWWWWWWWWW]  ← Kernel 执行

NVTX Range:                                        [=Data Proc==]  ← 标注区域

理想：PCIe 传输和 Kernel 执行应该尽量重叠（需要多 Stream）
```

### Nsight Compute Roofline 示意

```
性能 ^
(TOPS) │·····················  Tensor Core FP16 Roof (989 TOPS, H100)
       │            ·········  CUDA Core FP32 Roof (67 TFLOPS)
       │           /   ←Compute Bound
       │          / ✦ ← 优化后的 GEMM（接近 Tensor Core Roof）
       │         / ● ← 未优化的 MatMul（Compute Bound 但利用率低）
       │        /
       │---★---/     ← Memory-bound Kernel（如 Elementwise）
       │      /
       └──────────────────────→ 算术强度 (FLOPs/Byte)
```

---

## 🛠️ 动手练习

### 练习 1：Profile 昨天写的 Transpose Kernel

```bash
# 安装 Nsight Compute（如果没有）
apt-get install nsight-compute  # Ubuntu

# Profile transpose kernel
ncu --kernel-name "transpose_naive" \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    ./transpose

ncu --kernel-name "transpose_shared" \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    ./transpose

# 对比两个 Kernel 的：
# - Global Memory Throughput
# - L1 Hit Rate  
# - Warp Execution Efficiency
```

### 练习 2：用 nsys 分析数据传输瓶颈

```c
// profile_demo.cu：故意引入不必要的 H↔D 传输
#include <nvtx3/nvToolsExt.h>

int main() {
    // 场景A：每次迭代都传输（坏）
    nvtxRangePush("Bad: 多次传输");
    for (int i = 0; i < 100; i++) {
        cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
        kernel<<<grid, block>>>(d_data, n);
        cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    }
    nvtxRangePop();
    
    // 场景B：一次性传输（好）
    nvtxRangePush("Good: 一次传输");
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    for (int i = 0; i < 100; i++) {
        kernel<<<grid, block>>>(d_data, n);
    }
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    nvtxRangePop();
}
```

```bash
nsys profile --output=demo ./profile_demo
nsys stats demo.nsys-rep | grep -E "Kernel|Memcpy"
```

### 练习 3：PyTorch Profiler 分析 Transformer 层

```python
import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_ff=2048):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

model = SimpleTransformer().cuda()
x = torch.randn(32, 128, 512).cuda()  # batch=32, seq=128, d=512

from torch.profiler import profile, record_function, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    for _ in range(5):
        with record_function("transformer_forward"):
            y = model(x)
        torch.cuda.synchronize()

# 查看哪些算子最耗时
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
# 哪个最慢？Attention? LayerNorm? Linear?
```

---

## 📝 小结

| 工具 | 适合场景 | 关键输出 |
|------|---------|---------|
| `nvidia-smi` | 快速查看 GPU 负载 | 利用率、显存、功耗 |
| `nsys` | 找系统级瓶颈（传输/同步） | CPU-GPU 时间线 |
| `ncu` | 优化单个 Kernel | 内存/算力利用率、Roofline |
| `torch.profiler` | PyTorch 模型性能分析 | 算子耗时、内存分配 |

**工程原则**：
1. **先 nsys 找宏观瓶颈**（是传输慢还是 Kernel 慢？）
2. **再用 ncu 深入问题 Kernel**（是 Memory-bound 还是 Compute-bound？）
3. **对症下药**，而不是盲目猜测

**关键认知**：没有 Profile 数据的优化是"碰运气"。在 AI Infra 工作中，95% 的性能问题通过 nsys+ncu 就能定位。学会读懂 Roofline，是判断"这个 Kernel 还有多少优化空间"的核心能力。

---

**明日预告**：D7 是 Week1 的总结与综合实战——从零实现一个小型矩阵乘法 Kernel（SGEMM），应用本周所有知识，与 cuBLAS 对比性能。
