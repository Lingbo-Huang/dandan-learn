---
layout: default
title: "D3 · 成本优化：Spot 实例与资源调度"
render_with_liquid: false
---

# D3 · 成本优化：Spot 实例与资源调度

## GPU 训练成本计算

### 真实成本建模

```python
from dataclasses import dataclass

@dataclass
class TrainingCostModel:
    """LLM 训练成本估算模型"""
    
    # 硬件配置
    num_gpus: int = 1024
    gpu_type: str = "A100-80G"
    
    # 云厂商价格（$/GPU/小时）
    on_demand_price: float = 3.0   # 按需实例
    spot_price: float = 1.0        # Spot 实例（通常 30-40% 折扣）
    reserved_price: float = 2.0    # 1年预留实例
    
    # 训练参数
    total_tokens: float = 1e12     # 训练 token 数（1T tokens）
    tokens_per_second: float = 1e6  # 集群吞吐（1M tokens/s，理想状态）
    mfu: float = 0.45              # 实际 MFU
    
    def compute_training_time(self) -> float:
        """计算训练时间（小时）"""
        effective_tps = self.tokens_per_second * self.mfu
        total_seconds = self.total_tokens / effective_tps
        return total_seconds / 3600
    
    def compute_cost(self, use_spot: bool = False, spot_interruption_rate: float = 0.05):
        """
        计算总成本
        spot_interruption_rate: Spot 中断率（每小时中断概率）
        """
        training_hours = self.compute_training_time()
        
        if use_spot:
            # Spot 中断导致额外重算成本
            # 假设每次中断损失 checkpoint_interval 时间（默认 1000 步 ≈ 10 分钟）
            checkpoint_interval_hours = 10 / 60
            expected_interruptions = training_hours * spot_interruption_rate
            wasted_hours = expected_interruptions * checkpoint_interval_hours
            
            total_hours = training_hours + wasted_hours
            price = self.spot_price
        else:
            total_hours = training_hours
            price = self.on_demand_price
        
        gpu_cost = total_hours * self.num_gpus * price
        
        print(f"训练时间: {training_hours:.1f} 小时 ({training_hours/24:.1f} 天)")
        if use_spot:
            print(f"Spot 中断额外时间: {wasted_hours:.1f} 小时")
        print(f"GPU 成本: ${gpu_cost:,.0f}")
        print(f"单 GPU 小时: {total_hours:.1f}h × ${price}/h")
        
        return gpu_cost

# GPT-3 175B 训练成本估算
model = TrainingCostModel(
    num_gpus=1024,
    total_tokens=300e9,  # GPT-3 用了 300B tokens
    tokens_per_second=1536 * 2048 / 12.5,  # 1步12.5秒，batch=1536,seq=2048
    mfu=0.45
)

print("=== 按需实例 ===")
cost_on_demand = model.compute_cost(use_spot=False)

print("\n=== Spot 实例（中断率 5%/h）===")
cost_spot = model.compute_cost(use_spot=True, spot_interruption_rate=0.02)

print(f"\n节省: ${cost_on_demand - cost_spot:,.0f} ({(1-cost_spot/cost_on_demand):.0%})")
```

### GPU 利用率与成本的关系

```python
def roi_analysis(mfu: float, gpu_cost_per_hour: float, num_gpus: int):
    """
    ROI 分析：MFU 提升的价值
    """
    # 假设：训练固定 token 数
    # MFU 越高 → 训练时间越短 → 成本越低
    
    baseline_mfu = 0.3  # 典型基线
    target_tokens = 1e12
    
    # A100 FLOP 能力
    a100_peak_tflops = 312
    params = 70e9  # 70B 模型
    flops_per_token = 6 * params
    
    def tokens_per_second(mfu):
        return (mfu * a100_peak_tflops * 1e12 * num_gpus) / flops_per_token
    
    def training_hours(mfu):
        return target_tokens / tokens_per_second(mfu) / 3600
    
    def training_cost(mfu):
        return training_hours(mfu) * num_gpus * gpu_cost_per_hour
    
    baseline_cost = training_cost(baseline_mfu)
    target_cost = training_cost(mfu)
    
    savings = baseline_cost - target_cost
    
    print(f"MFU {baseline_mfu:.0%} → {mfu:.0%} 的价值:")
    print(f"  训练时间: {training_hours(baseline_mfu):.1f}h → {training_hours(mfu):.1f}h")
    print(f"  训练成本: ${baseline_cost:,.0f} → ${target_cost:,.0f}")
    print(f"  节省: ${savings:,.0f} ({savings/baseline_cost:.0%})")

# 将 MFU 从 30% 提升到 45%
roi_analysis(mfu=0.45, gpu_cost_per_hour=3.0, num_gpus=256)
# 输出：节省约 33%，对于百万美元级训练来说非常可观
```

## Spot 实例策略

```python
import boto3
import time

class SpotTrainingManager:
    """
    AWS Spot Instance 管理
    在中断时自动迁移到新实例继续训练
    """
    
    def __init__(self, region="us-east-1"):
        self.ec2 = boto3.client("ec2", region_name=region)
        self.instance_interruption_warned = False
    
    def check_spot_interruption(self) -> bool:
        """
        检查 Spot 中断警告
        AWS 会在中断前 2 分钟发出通知
        """
        try:
            import urllib.request
            # EC2 instance metadata 服务
            url = "http://169.254.169.254/latest/meta-data/spot/termination-time"
            req = urllib.request.urlopen(url, timeout=1)
            # 如果能访问到，说明 Spot 即将被中断
            return True
        except:
            return False
    
    def handle_interruption(self, checkpointer, step, model, optimizer, scheduler, loss):
        """
        收到中断信号时立即保存 checkpoint
        """
        print(f"⚠️ Spot 中断警告！立即保存 checkpoint (step={step})")
        
        # 同步保存（不异步，确保完成）
        checkpointer.save(
            step=step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
        )
        
        # 通知调度系统（如 Kubernetes Job 或自定义 controller）
        self._notify_scheduler("interrupted", step)
        
        print("Checkpoint 保存完成，等待实例终止")
    
    def _notify_scheduler(self, status: str, step: int):
        """通知外部调度系统"""
        import requests
        try:
            requests.post(
                "http://training-scheduler/api/status",
                json={"status": status, "step": step},
                timeout=5
            )
        except:
            pass  # 调度系统可能也受影响

    def training_loop_with_spot(self, model, optimizer, dataloader,
                                 checkpointer, max_steps):
        """集成 Spot 中断处理的训练循环"""
        step = checkpointer.load_latest(model, optimizer, None)
        
        for batch in dataloader:
            # 每 100 步检查一次中断警告
            if step % 100 == 0:
                if self.check_spot_interruption():
                    self.handle_interruption(...)
                    return  # 优雅退出
            
            loss = train_step(model, optimizer, batch)
            step += 1
            
            if step >= max_steps:
                break
```

## GPU 资源调度：混合训练策略

```python
"""
Gang Scheduling：所有 rank 必须同时获得资源
Elastic Training：允许 rank 数量动态变化
"""

# 弹性训练（PyTorch Elastic）
# torchrun 支持节点动态加入/退出

# 使用 torchrun 启动弹性训练
# torchrun --min-nodes=8 --max-nodes=16 --nproc-per-node=8 train.py

import torch.distributed.elastic.multiprocessing as mp

def setup_elastic_training():
    """弹性训练初始化"""
    from torch.distributed.elastic.rendezvous import RendezvousParameters
    from torch.distributed.elastic.agent.server import SimpleElasticAgent
    
    # 最少 8 节点，最多 16 节点
    # 如果节点数低于 8，等待更多节点加入
    # 如果有节点失败，只要还有 8+ 节点就继续
    pass

# 资源调度优先级
class GPUClusterScheduler:
    """GPU 集群任务调度"""
    
    def __init__(self):
        self.queue = []      # 待调度的训练任务
        self.running = []    # 运行中的任务
        self.gpu_pool = []   # 可用 GPU
    
    def schedule(self):
        """
        调度策略：
        1. 优先保障大规模任务（避免碎片化）
        2. Spot 实例优先用于可中断的实验
        3. 预留实例用于生产训练
        """
        for task in sorted(self.queue, key=lambda t: -t.priority):
            available_gpus = self._find_available_gpus(task.num_gpus)
            if available_gpus:
                self._launch_task(task, available_gpus)
    
    def _find_available_gpus(self, n: int) -> list:
        """
        寻找满足数量要求的 GPU
        注意：分布式训练需要 GPU 在同一局域网内（低延迟）
        """
        # 优先从同一机架/POD 内选取（减少跨交换机流量）
        # ... 实现拓扑感知的 GPU 分配
        pass
```

## 成本优化最佳实践

| 策略 | 节省幅度 | 难度 | 适用场景 |
|------|---------|------|---------|
| Spot 实例 | 60-70% | 中 | 有 checkpoint 的长训练 |
| 预留实例 | 30-40% | 低 | 稳定的生产训练 |
| 提升 MFU | 20-50% | 高 | 需要优化并行配置 |
| 混合精度 BF16 | 10-20% | 低 | 几乎所有场景 |
| 激活 checkpointing | -10%（速度损失）| 低 | 内存受限场景 |
| 梯度累积 | 减少 GPU 数 | 低 | 小规模试验 |
| 弹性训练 | 15-25% | 高 | 支持节点动态扩缩 |

## 面试题

**Q: 如何评估 GPU 集群的利用率，并提出优化方案？**

A: 评估维度：①**MFU（Model FLOP Utilization）**：实际计算量/理论峰值，目标 >35%；②**GPU Utilization**：SM 使用率，应 >90%；③**HBM Bandwidth Utilization**：实际 vs 峰值带宽；④**通信/计算比**：NCCL 通信占总时间的比例。常见瓶颈及方案：**数据瓶颈**（GPU Util 低、IO 高）→ 预取数据、增加 DataLoader 线程数；**通信瓶颈**（DP All-Reduce 占比高）→ 梯度压缩、异步通信、优化 TP/PP 比例；**内存瓶颈**（频繁 OOM 或 recompute）→ 调整并行度、优化批大小、使用梯度 checkpoint；**调度碎片**（部分 GPU 空闲）→ 改用 Gang Scheduling + 弹性训练。
