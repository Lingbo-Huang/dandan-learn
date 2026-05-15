---
layout: default
title: "D2 · 故障恢复：Checkpoint 策略与自动重启"
render_with_liquid: false
---

# D2 · 故障恢复：Checkpoint 策略与自动重启

## 大规模训练的故障率

**硬件故障率**（经验数据）：
- 单 GPU 故障率：~0.1%/天
- 1024 GPU 集群：预期每 1-2 天发生 1 次 GPU 故障
- 一周训练（168小时）：50% 概率遇到至少一次硬件故障

**结论**：不是"是否会故障"，而是"何时故障"。Checkpoint 是生命线。

## Checkpoint 策略设计

### 基础：分布式 Checkpoint

```python
import torch
import torch.distributed as dist
from pathlib import Path
import time
import json

class DistributedCheckpointer:
    """
    分布式训练 Checkpoint 管理器
    支持：异步保存、版本轮转、完整性验证
    """
    
    def __init__(
        self,
        save_dir: str,
        rank: int,
        world_size: int,
        keep_last_n: int = 3,          # 保留最近 N 个 checkpoint
        async_save: bool = True,        # 异步保存（不阻塞训练）
    ):
        self.save_dir = Path(save_dir)
        self.rank = rank
        self.world_size = world_size
        self.keep_last_n = keep_last_n
        self.async_save = async_save
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss: float,
        extra_state: dict = None
    ):
        """保存 checkpoint"""
        start_time = time.time()
        
        # 每个 rank 保存自己的分片
        ckpt_dir = self.save_dir / f"step_{step:08d}"
        ckpt_dir.mkdir(exist_ok=True)
        
        # 模型状态（TP/PP 分片）
        rank_ckpt_path = ckpt_dir / f"model_rank_{self.rank:04d}.pt"
        
        state = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "loss": loss,
            "rank": self.rank,
            "world_size": self.world_size,
        }
        if extra_state:
            state.update(extra_state)
        
        # 异步保存（不阻塞训练主循环）
        if self.async_save:
            import threading
            def _save():
                torch.save(state, rank_ckpt_path)
                if self.rank == 0:
                    self._write_metadata(ckpt_dir, step, loss)
            
            t = threading.Thread(target=_save)
            t.start()
            # 注意：下一次 save 前需要确认上次完成
            return t
        else:
            torch.save(state, rank_ckpt_path)
            if self.rank == 0:
                self._write_metadata(ckpt_dir, step, loss)
        
        elapsed = time.time() - start_time
        if self.rank == 0:
            print(f"Checkpoint saved: step={step}, time={elapsed:.1f}s")
        
        # 清理旧 checkpoint
        if self.rank == 0:
            self._cleanup_old_checkpoints()
    
    def _write_metadata(self, ckpt_dir: Path, step: int, loss: float):
        """写入元数据（用于验证完整性）"""
        metadata = {
            "step": step,
            "loss": loss,
            "world_size": self.world_size,
            "timestamp": time.time(),
            "complete": True,  # 所有 rank 写完后标记
        }
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
    
    def _cleanup_old_checkpoints(self):
        """保留最近 N 个，删除旧的"""
        checkpoints = sorted(self.save_dir.glob("step_*"), key=lambda p: p.name)
        if len(checkpoints) > self.keep_last_n:
            for old_ckpt in checkpoints[:-self.keep_last_n]:
                import shutil
                shutil.rmtree(old_ckpt)
                print(f"Removed old checkpoint: {old_ckpt}")
    
    def load_latest(self, model, optimizer, scheduler):
        """加载最新的完整 checkpoint"""
        checkpoints = sorted(self.save_dir.glob("step_*"))
        
        for ckpt_dir in reversed(checkpoints):
            # 验证完整性
            metadata_path = ckpt_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            if not metadata.get("complete"):
                print(f"Skipping incomplete checkpoint: {ckpt_dir}")
                continue
            
            # 加载本 rank 的状态
            rank_ckpt = ckpt_dir / f"model_rank_{self.rank:04d}.pt"
            if not rank_ckpt.exists():
                print(f"Missing rank file: {rank_ckpt}")
                continue
            
            state = torch.load(rank_ckpt, map_location="cpu")
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            if scheduler and state.get("scheduler_state_dict"):
                scheduler.load_state_dict(state["scheduler_state_dict"])
            
            print(f"Loaded checkpoint from step {state[step]}, loss={state[loss]:.4f}")
            return state["step"]
        
        print("No valid checkpoint found, starting from scratch")
        return 0
```

### 进阶：Checkpoint 频率策略

```python
class AdaptiveCheckpointScheduler:
    """
    自适应 Checkpoint 调度
    - 初期：较少保存（训练不稳定时频繁）
    - 稳定后：按固定间隔保存
    - 重要里程碑：额外保存
    """
    
    def __init__(
        self,
        base_interval: int = 1000,   # 基础保存间隔
        warmup_interval: int = 100,  # 预热期保存间隔
        warmup_steps: int = 1000,    # 预热步数
    ):
        self.base_interval = base_interval
        self.warmup_interval = warmup_interval
        self.warmup_steps = warmup_steps
        self.loss_spikes = []         # 记录 loss spike 的步数
    
    def should_save(self, step: int, loss: float, loss_history: list) -> bool:
        # 预热期更频繁保存
        if step < self.warmup_steps:
            return step % self.warmup_interval == 0
        
        # 普通区间
        if step % self.base_interval == 0:
            return True
        
        # Loss Spike 后立即保存（其实是"保存 spike 前的版本"）
        if len(loss_history) >= 2:
            ratio = loss / (loss_history[-2] + 1e-8)
            if ratio > 1.3:
                self.loss_spikes.append(step)
                # 这时应该考虑回滚，而不是保存
                return False
        
        return False
    
    def should_rollback(self, step: int, loss: float, loss_history: list) -> bool:
        """判断是否需要回滚到上一个 checkpoint"""
        if len(loss_history) < 10:
            return False
        
        # 连续 5 步 loss 上升 + 当前 loss 比最近 10 步均值高 50%
        recent_avg = sum(loss_history[-10:]) / 10
        if loss > recent_avg * 1.5:
            consecutive_increase = all(
                loss_history[-i] > loss_history[-i-1]
                for i in range(1, 6)
                if i+1 < len(loss_history)
            )
            if consecutive_increase:
                return True
        
        return False
```

## 自动故障恢复

```bash
#!/bin/bash
# train_with_resume.sh - 带自动重启的训练脚本

MAX_RETRIES=10
RETRY_DELAY=60  # 等待 60 秒后重试（给系统恢复时间）
retry_count=0

while [ $retry_count -lt $MAX_RETRIES ]; do
    echo "[$(date)] 启动训练，第 $((retry_count + 1)) 次尝试"
    
    # 运行训练
    torchrun \
        --nproc_per_node=8 \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=29500 \
        train.py \
        --resume_from_checkpoint latest \
        --checkpoint_dir /ckpt/run-001 \
        "$@"
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "训练正常完成"
        exit 0
    fi
    
    # 分析退出码
    case $exit_code in
        1)   echo "程序错误，停止重试"; exit 1 ;;
        139) echo "Segfault（GPU 硬件问题？），等待 ${RETRY_DELAY}s 后重试" ;;
        -9)  echo "OOM Kill，等待后重试" ;;
        *)   echo "未知错误 (exit=$exit_code)，等待后重试" ;;
    esac
    
    retry_count=$((retry_count + 1))
    
    # 发送告警
    curl -X POST https://hooks.slack.com/services/xxx \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"训练中断 (exit=$exit_code)，正在重试 ($retry_count/$MAX_RETRIES)\"}"
    
    sleep $RETRY_DELAY
done

echo "达到最大重试次数，放弃"
exit 1
```

```python
# Python 层面的健壮训练循环
def robust_train_loop(
    model, optimizer, scheduler, dataloader,
    checkpointer: DistributedCheckpointer,
    ckpt_scheduler: AdaptiveCheckpointScheduler,
    max_steps: int,
):
    """带异常处理的训练主循环"""
    
    # 断点续训
    start_step = checkpointer.load_latest(model, optimizer, scheduler)
    
    loss_history = []
    step_time_buffer = None  # 上一次的异步 save 任务
    
    for step, batch in enumerate(dataloader, start=start_step):
        if step >= max_steps:
            break
        
        # 等待上一次异步 checkpoint 完成
        if step_time_buffer is not None:
            step_time_buffer.join()
            step_time_buffer = None
        
        try:
            step_start = time.time()
            loss = train_step(model, optimizer, scheduler, batch)
            step_time = time.time() - step_start
            
            loss_val = loss.item()
            loss_history.append(loss_val)
            
            # 检查 NaN
            if not torch.isfinite(torch.tensor(loss_val)):
                print(f"Step {step}: Loss is {loss_val}, loading last checkpoint")
                checkpointer.load_latest(model, optimizer, scheduler)
                continue
            
            # 检查是否需要保存
            if ckpt_scheduler.should_save(step, loss_val, loss_history):
                step_time_buffer = checkpointer.save(
                    step, model, optimizer, scheduler, loss_val
                )
            
            # 检查是否需要回滚
            if ckpt_scheduler.should_rollback(step, loss_val, loss_history):
                print(f"Step {step}: Detected training instability, rolling back")
                start_step = checkpointer.load_latest(model, optimizer, scheduler)
                loss_history = loss_history[:start_step]
        
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                print(f"OOM at step {step}, skipping batch")
                continue
            elif "NCCL" in str(e):
                print(f"NCCL error at step {step}: {e}")
                raise  # 触发外部重启
            else:
                raise
```

## 面试题

**Q: Checkpoint 保存时，异步保存的风险是什么？如何规避？**

A: 异步保存的主要风险：①**数据一致性**：保存期间模型继续训练，如果下一次保存触发时上次还没完成，可能保存中间状态（用 join() 确保完成）；②**磁盘写入不完整**：进程崩溃时 checkpoint 文件可能不完整（用 metadata.json 标记"完整"标志，加载时检查）；③**多 rank 不同步**：部分 rank 写完、部分未完成（用 dist.barrier() 同步所有 rank 后才标记 complete）。最佳实践：写入临时目录，全部完成后原子重命名到正式路径；或使用 PyTorch 原生的 `torch.distributed.checkpoint` (DCP) 支持可靠的分布式 checkpoint。
