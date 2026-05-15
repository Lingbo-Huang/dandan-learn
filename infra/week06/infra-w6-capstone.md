---
layout: default
title: "Week 6 Capstone · 设计千卡 LLM 训练系统"
render_with_liquid: false
---

# Week 6 Capstone · 设计千卡 LLM 训练系统

## 系统设计题目

**题目**：设计一个用于训练 GPT-3 级别（175B 参数）语言模型的分布式训练系统。

**约束条件**：
- 硬件：1024 张 A100 80GB（每节点 8 卡，共 128 节点）
- 目标：尽量高效地训练，避免 OOM，最大化 MFU（Model FLOP Utilization）
- 序列长度：2048 tokens
- 全局 batch size：1536（符合 GPT-3 论文）

## 第一步：并行配置选择

### 单 GPU 内存分析

GPT-3 175B 参数占用：
```python
# 混合精度训练的内存占用
params = 175e9

fp16_weights    = params * 2        # 334 GB
fp32_weights    = params * 4        # 700 GB (Adam master weights)
fp32_gradients  = params * 4        # 700 GB
adam_m          = params * 4        # 700 GB
adam_v          = params * 4        # 700 GB

total_optimizer = fp32_weights + fp32_gradients + adam_m + adam_v  # 2800 GB
total = fp16_weights + total_optimizer  # ~3134 GB

print(f"单卡内存需求: {total/1e9:.0f} GB（A100 只有 80GB！）")
# → 必须使用大规模并行
```

### 并行配置搜索

```python
"""
并行配置最优化
约束：TP × PP × DP = 1024
"""
import math

def estimate_efficiency(tp, pp, dp, num_microbatches_per_global):
    """估算训练效率"""
    # TP 通信代价（NVLink 节点内）
    tp_comm_overhead = 0.05 * (tp - 1) / tp  # 假设 5% 通信占比/节点
    
    # PP Bubble 率
    m = num_microbatches_per_global
    bubble_rate = (pp - 1) / (m + pp - 1)
    
    # 有效 MFU
    mfu = (1 - tp_comm_overhead) * (1 - bubble_rate)
    
    return mfu

def check_memory_ok(tp, pp, dp, global_batch=1536, seq_len=2048):
    """检查内存是否足够"""
    num_layers = 96
    hidden_size = 12288
    
    # 每个 PP Stage 的层数
    layers_per_stage = num_layers // pp
    
    # 每个 rank 的模型大小（FP16）
    params_per_rank = 175e9 / (tp * pp)  # 权重被 TP 和 PP 切分
    weights_gb = params_per_rank * 2 / 1e9
    
    # 优化器状态（FP32，用 ZeRO-1 每 DP rank 只存 1/DP）
    optimizer_gb = (params_per_rank / dp) * 12 / 1e9  # 3 × FP32
    
    # 激活值（1F1B：存 pp 个 microbatch）
    microbatch_size = global_batch // dp
    num_microbatches = microbatch_size  # 假设微批大小=1
    activations_per_layer = microbatch_size * seq_len * hidden_size * 2 / 1e9
    activations_gb = activations_per_layer * layers_per_stage * min(pp, num_microbatches)
    
    total_gb = weights_gb + optimizer_gb + activations_gb
    
    return total_gb < 75, total_gb  # 留 5GB buffer

# 搜索最优配置
best_config = None
best_mfu = 0

for tp in [1, 2, 4, 8]:
    for pp in [1, 2, 4, 8, 16]:
        dp = 1024 // (tp * pp)
        if dp < 1:
            continue
        if tp * pp * dp != 1024:
            continue
        
        # TP 最好在节点内（8 卡/节点）
        if tp > 8:
            continue
        
        microbatch_size = 1536 // dp  # 假设微批大小 = 全局BS / DP
        if microbatch_size < 1:
            continue
        
        ok, mem = check_memory_ok(tp, pp, dp)
        if not ok:
            continue
        
        m = microbatch_size
        mfu = estimate_efficiency(tp, pp, dp, m)
        
        print(f"TP={tp} PP={pp} DP={dp}: "
              f"mem={mem:.1f}GB, MFU≈{mfu:.2%}, m={m}")
        
        if mfu > best_mfu:
            best_mfu = mfu
            best_config = (tp, pp, dp)

print(f"\n最优配置: TP={best_config[0]}, PP={best_config[1]}, DP={best_config[2]}")
print(f"预估 MFU: {best_mfu:.2%}")

# 典型答案：TP=8, PP=16, DP=8（充分利用节点内 NVLink）
# 或 TP=8, PP=8, DP=16（PP Bubble 小，DP 梯度通信多）
```

### 推荐配置

**TP=8（节点内）, PP=8, DP=16**：

```
1024 GPU = 8 (TP/节点) × 8 (PP/TP组) × 16 (DP)

内存分析（每 rank）：
  模型权重：175B / (8×8) × 2B = 5.5 GB
  优化器：175B / (8×8) × 12B / 16 = 1.7 GB
  激活值（1F1B, 8 microbatch）：~20 GB
  总计：~27 GB ✓（< 80 GB）

Bubble 率（m=12, pp=8）：
  (8-1)/(12+8-1) = 7/19 ≈ 37%（需要增加 m）
  
  实际设置 m = 64（global_batch=1536, dp=16, microbatch=1）
  (8-1)/(64+8-1) = 7/71 ≈ 10% ✓
```

## 第二步：网络拓扑规划

```
节点内（8× A100，NVLink 600 GB/s）：
  TP=8 (Tensor Parallel) ← 最高通信密度

跨节点（InfiniBand 8× HDR，约 200 Gbps = 25 GB/s per 节点）：
  PP=8 (Pipeline Parallel) ← 点对点 P2P，通信量适中
  DP=16 (Data Parallel)   ← All-Reduce，用 ZeRO-1 优化

节点分配：
  每个 DP 组 = 8 (TP) × 8 (PP) = 64 GPU = 8 节点
  16 个 DP 组 = 16 × 8 节点 = 128 节点 ✓
```

## 第三步：训练吞吐量估算

```python
def estimate_throughput():
    # 硬件参数
    peak_flops = 312e12   # A100 FP16 Tensor Core
    hbm_bw     = 2e12     # A100 HBM 带宽
    nvlink_bw  = 600e9    # NVLink 双向带宽
    ib_bw      = 25e9     # InfiniBand per 节点（8× HDR）
    
    # 模型参数
    N, d = 96, 12288      # 层数, 隐层维度
    seq_len = 2048
    
    # 每个 rank 的 FLOPs（前向 + 反向 ≈ 3× 前向）
    # 每 token 的 FLOPs（近似）= 6 × 参数量（前向 + 反向）
    params_per_rank = 175e9 / (8 * 8)  # TP=8, PP=8
    tokens_per_step = 1536 * 2048       # global_batch × seq_len
    
    # 理论峰值时间
    flops_per_step = 6 * 175e9 * tokens_per_step
    time_compute = flops_per_step / (1024 * peak_flops)
    
    # 通信时间（粗略估计）
    # TP All-Reduce：每层 2 次，每次 [1, 2048, 12288] × 2B
    tp_data = 2 * N * 1 * 2048 * 12288 * 2  # bytes
    time_tp = tp_data / nvlink_bw
    
    # PP P2P：每个 boundary 一次 [1, 2048, 12288] × 2B
    pp_boundaries = 8 - 1
    pp_data = pp_boundaries * 1 * 2048 * 12288 * 2
    time_pp = pp_data / ib_bw
    
    # DP ZeRO-1 All-Reduce（梯度通信）
    grad_data = 175e9 * 2 / 16  # FP16, ZeRO-1
    time_dp = grad_data / ib_bw
    
    # 实际时间（计算与通信并行，取 max）
    total_time = time_compute * (1 + 0.37)  # Bubble 率 37%
    
    print(f"计算时间：{time_compute:.2f}s/step")
    print(f"TP 通信：{time_tp:.3f}s")
    print(f"PP 通信：{time_pp:.3f}s")
    print(f"DP 通信：{time_dp:.2f}s（与计算重叠）")
    print(f"总时间（含 Bubble）：{total_time:.2f}s/step")
    
    # MFU
    mfu = flops_per_step / (1024 * peak_flops * total_time)
    print(f"理论 MFU：{mfu:.1%}")

estimate_throughput()
```

## 第四步：训练稳定性策略

```yaml
# 训练配置（YAML 风格）
training:
  # 优化器
  optimizer: adam
  adam_beta1: 0.9
  adam_beta2: 0.95
  weight_decay: 0.1
  
  # 学习率调度
  lr_schedule: cosine
  warmup_iters: 375
  max_lr: 6e-5
  min_lr: 6e-6
  
  # 梯度裁剪（防止梯度爆炸）
  grad_clip: 1.0
  
  # 数值稳定性
  bf16: true             # BF16 比 FP16 更稳定（范围大）
  loss_scale: dynamic    # 动态损失缩放（针对 FP16）
  
  # 检查点
  checkpoint_interval: 1000  # 每 1000 步保存
  checkpoint_activations: true  # Gradient Checkpointing（节省内存）
  
  # 监控
  log_throughput: true
  log_memory: true
  spike_threshold: 1.5   # loss spike 超过 1.5× 则回滚
```

## 本周总结

| 技术 | 作用 | 关键参数 |
|------|------|---------|
| Tensor Parallel | 切分权重矩阵 | TP=8（节点内 NVLink）|
| Pipeline Parallel | 切分模型层数 | PP=4-16（Bubble vs 内存） |
| Data Parallel | 扩展 batch size | DP 越大吞吐越高 |
| 序列并行 | 减少激活值内存 | 随 TP 同步开启 |
| 1F1B 调度 | 降低 PP 内存峰值 | 激活 O(p)→O(pp) |
| ZeRO-1/2/3 | 减少优化器状态内存 | 配合 DP 使用 |
| MoE | 提升参数效率 | EP 与 TP/PP 组合 |

**下周预告：量化与压缩——INT8/INT4/AWQ/GPTQ/SmoothQuant**
