---
layout: default
title: "D1 · 训练监控：指标、告警与异常检测"
render_with_liquid: false
---

# D1 · 训练监控：指标、告警与异常检测

## 为什么监控是 AI Infra 的核心技能

千卡集群一次训练可能跑几周，耗资数百万元。没有完善的监控，一个小问题可能悄悄浪费几十小时的计算资源。

## 核心监控指标体系

### 1. 训练健康指标

```python
import torch
import wandb
import time
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class TrainingMetrics:
    """训练监控指标收集器"""
    
    # 模型指标
    loss_history: List[float] = field(default_factory=list)
    grad_norm_history: List[float] = field(default_factory=list)
    lr_history: List[float] = field(default_factory=list)
    
    # 性能指标
    step_time_history: List[float] = field(default_factory=list)
    tokens_per_second_history: List[float] = field(default_factory=list)
    
    # 硬件指标
    gpu_util_history: List[float] = field(default_factory=list)
    gpu_mem_history: List[float] = field(default_factory=list)
    
    def log_step(self, step: int, loss: float, grad_norm: float,
                 lr: float, step_time: float, batch_tokens: int,
                 gpu_rank: int = 0):
        """记录一步训练的所有指标"""
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        self.lr_history.append(lr)
        self.step_time_history.append(step_time)
        
        tps = batch_tokens / step_time
        self.tokens_per_second_history.append(tps)
        
        # 获取 GPU 指标
        if torch.cuda.is_available():
            gpu_util = torch.cuda.utilization(gpu_rank)
            gpu_mem = torch.cuda.memory_allocated(gpu_rank) / 1e9
            self.gpu_util_history.append(gpu_util)
            self.gpu_mem_history.append(gpu_mem)
        
        # 上报到 W&B
        if wandb.run:
            wandb.log({
                "train/loss": loss,
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "train/step_time": step_time,
                "train/tokens_per_second": tps,
                "system/gpu_utilization": gpu_util if torch.cuda.is_available() else 0,
                "system/gpu_memory_gb": gpu_mem if torch.cuda.is_available() else 0,
            }, step=step)
    
    def detect_anomalies(self, window: int = 10) -> Dict[str, str]:
        """异常检测"""
        alerts = {}
        
        if len(self.loss_history) < window + 1:
            return alerts
        
        recent_loss = self.loss_history[-window:]
        prev_loss = self.loss_history[-(window*2):-window]
        
        # 1. Loss Spike：单步 loss 突增 > 50%
        if len(self.loss_history) >= 2:
            loss_ratio = self.loss_history[-1] / (self.loss_history[-2] + 1e-8)
            if loss_ratio > 1.5:
                alerts["loss_spike"] = f"Loss spike: {loss_ratio:.1f}× ({self.loss_history[-2]:.3f} → {self.loss_history[-1]:.3f})"
        
        # 2. Loss 持续不降：最近 window 步平均没有改善
        if prev_loss:
            if sum(recent_loss)/len(recent_loss) >= sum(prev_loss)/len(prev_loss) * 0.99:
                alerts["loss_plateau"] = f"Loss plateau: {sum(recent_loss)/len(recent_loss):.4f}"
        
        # 3. 梯度爆炸：grad_norm > 100
        if self.grad_norm_history and self.grad_norm_history[-1] > 100:
            alerts["grad_explosion"] = f"Grad norm: {self.grad_norm_history[-1]:.1f}"
        
        # 4. 梯度消失：grad_norm < 1e-6
        if self.grad_norm_history and self.grad_norm_history[-1] < 1e-6:
            alerts["grad_vanish"] = f"Grad norm: {self.grad_norm_history[-1]:.2e}"
        
        # 5. GPU 利用率低：< 80%（可能有数据瓶颈）
        if self.gpu_util_history and self.gpu_util_history[-1] < 80:
            alerts["low_gpu_util"] = f"GPU util: {self.gpu_util_history[-1]:.1f}% (bottleneck?)"
        
        # 6. 吞吐量下降：比历史均值低 20%
        if len(self.tokens_per_second_history) > window:
            avg_tps = sum(self.tokens_per_second_history[:-window]) / (len(self.tokens_per_second_history) - window)
            recent_tps = self.tokens_per_second_history[-1]
            if recent_tps < avg_tps * 0.8:
                alerts["throughput_drop"] = f"TPS: {recent_tps:.0f} (avg: {avg_tps:.0f})"
        
        return alerts
```

### 2. MFU（Model FLOP Utilization）计算

MFU 是衡量训练效率的黄金指标：

```python
def compute_mfu(
    model_params: int,          # 模型参数量
    batch_tokens: int,          # 本 step 处理的 token 数
    step_time_seconds: float,   # step 耗时（秒）
    num_gpus: int,              # GPU 数量
    gpu_flops_peak: float = 312e12  # A100 FP16 峰值算力（312 TFLOPS）
) -> float:
    """
    计算 Model FLOP Utilization (MFU)
    
    实际 FLOPs = 6 × 参数量 × token 数
    （前向 2N × tokens + 反向 4N × tokens ≈ 6N × tokens）
    
    MFU = 实际 FLOPs / (GPU 数 × 峰值算力 × 时间)
    """
    actual_flops = 6 * model_params * batch_tokens
    theoretical_flops = num_gpus * gpu_flops_peak * step_time_seconds
    mfu = actual_flops / theoretical_flops
    return mfu

# 示例：GPT-3 175B 训练
mfu = compute_mfu(
    model_params=175e9,
    batch_tokens=1536 * 2048,   # batch_size × seq_len
    step_time_seconds=12.5,     # 实测每步耗时
    num_gpus=1024,
    gpu_flops_peak=312e12
)
print(f"MFU: {mfu:.1%}")  # 典型值：30-50%（好的系统）

# MFU 参考值：
# < 20%: 存在严重瓶颈（数据加载、通信问题）
# 20-35%: 一般（通信开销大，或 PP Bubble 严重）
# 35-50%: 较好（优化后的分布式训练）
# > 50%: 优秀（接近硬件极限）
```

### 3. NCCL 通信监控

```python
import subprocess
import re

def monitor_nccl_bandwidth(rank: int, duration_seconds: int = 10):
    """
    监控 NCCL 通信带宽
    通过 NCCL_DEBUG=INFO 日志解析
    """
    env = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "NET"}
    
    # 解析 NCCL 日志中的带宽信息
    # 格式: NCCL INFO AllReduce: Bandwidth=XX GB/s
    pattern = re.compile(r"AllReduce.*Bandwidth=(\d+\.?\d*)\s*GB/s")
    
    bandwidths = []
    with open(f"/tmp/nccl_rank{rank}.log") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                bandwidths.append(float(m.group(1)))
    
    if bandwidths:
        print(f"Rank {rank} NCCL AllReduce 带宽:")
        print(f"  均值: {sum(bandwidths)/len(bandwidths):.1f} GB/s")
        print(f"  最小: {min(bandwidths):.1f} GB/s")
        print(f"  最大: {max(bandwidths):.1f} GB/s")
        
        # 诊断
        if min(bandwidths) < 50:  # IB 带宽应 > 100 GB/s
            print("  ⚠️  带宽异常低，可能有网络问题或掉队节点")
    
    return bandwidths


def detect_straggler_node(all_rank_step_times: dict) -> list:
    """
    检测掉队节点（Straggler）
    分布式训练中，最慢的 rank 决定整体速度
    
    all_rank_step_times: {rank: [step_times]}
    """
    avg_times = {rank: sum(times)/len(times) for rank, times in all_rank_step_times.items()}
    global_avg = sum(avg_times.values()) / len(avg_times)
    
    stragglers = []
    for rank, avg_time in avg_times.items():
        if avg_time > global_avg * 1.1:  # 比均值慢 10%+
            stragglers.append({
                "rank": rank,
                "avg_step_time": avg_time,
                "slowdown": avg_time / global_avg - 1,
            })
    
    if stragglers:
        for s in stragglers:
            print(f"⚠️ 掉队节点 Rank {s[rank]}: "
                  f"慢 {s[slowdown]:.1%}，avg_time={s[avg_step_time]:.2f}s")
    
    return stragglers
```

## 训练监控仪表板（Prometheus + Grafana）

```yaml
# prometheus.yml - 抓取 AI 训练指标
scrape_configs:
  - job_name: gpu_metrics
    scrape_interval: 10s
    static_configs:
      - targets: [node-01:9400, node-02:9400, ...]
    # 使用 nvidia-smi exporter 暴露 GPU 指标

  - job_name: training_metrics
    scrape_interval: 5s
    static_configs:
      - targets: [training-server:8080]  # 自定义 metrics 端点
```

```python
# 自定义 Prometheus 指标端点
from prometheus_client import start_http_server, Gauge, Counter

# 定义指标
TRAINING_LOSS = Gauge("training_loss", "Current training loss")
TRAINING_MFU = Gauge("training_mfu", "Model FLOP Utilization", ["rank"])
GPU_UTIL = Gauge("gpu_utilization", "GPU utilization %", ["node", "gpu_id"])
TOKENS_PER_SEC = Gauge("tokens_per_second", "Training throughput")
GRAD_NORM = Gauge("gradient_norm", "Gradient norm")

def update_metrics(step_info: dict):
    TRAINING_LOSS.set(step_info["loss"])
    TRAINING_MFU.labels(rank=step_info["rank"]).set(step_info["mfu"])
    GPU_UTIL.labels(
        node=step_info["node"],
        gpu_id=step_info["gpu_id"]
    ).set(step_info["gpu_util"])
    TOKENS_PER_SEC.set(step_info["tps"])
    GRAD_NORM.set(step_info["grad_norm"])

# 启动 metrics 服务
start_http_server(8080)
```

## 告警规则

```yaml
# Grafana AlertManager 规则
groups:
  - name: training_alerts
    rules:
      - alert: LossSpike
        expr: training_loss > training_loss offset 5m * 1.5
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "训练 Loss 突增 50%，检查是否需要回滚"
      
      - alert: LowMFU
        expr: avg(training_mfu) < 0.2
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "MFU < 20%，存在严重性能瓶颈"
      
      - alert: GPUMemoryHigh
        expr: gpu_memory_used_percent > 95
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "GPU 显存即将耗尽（OOM 风险）"
      
      - alert: StragglerNode
        expr: max(step_time_seconds) / avg(step_time_seconds) > 1.15
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "检测到掉队节点，影响整体训练效率"
```

## 面试题

**Q: 训练中 Loss NaN 怎么排查？**

A: Loss NaN 的根本原因是某处出现了数值溢出。排查步骤：①检查梯度裁剪是否生效（grad_norm 是否 < 1）；②检查学习率是否过大（突然 NaN 往往是 LR spike）；③检查是否有 log(0) 或除以零（如 cross_entropy 时 logit 全为 -inf）；④用 `torch.autograd.detect_anomaly()` 定位首次出现 NaN 的层；⑤切换 FP32 训练验证是否是精度问题（FP16 动态范围小）；⑥检查数据中是否有异常值（缺失 token、超长序列）。最常见原因：动态损失缩放（loss scale）溢出，解决方法是降低 loss_scale 初始值或增大 loss_scale_window。
